"""Evaluate a trained GraphLevelGraphPFN checkpoint (from
bin/graphpfn/pretrain_graph_level.py) via zero-gradient ICL on real QM9
molecules, at a context size and target selection independent of whatever
[base_config.qm9_eval] happened to be pinned to during training.

Unlike dev/pool_icl_real_limix.py / dev/qm9_pool_icl_real_limix.py (both
predate GraphLevelGraphPFN and manually replicate FeaturesTransformer.forward
by hand), this loads the real trained model and calls it exactly as
pretrain_graph_level.py's own eval_fn does -- same GraphLevelGraphPFN.forward,
same context_query_split/standardize_by_context conventions, same "QM9 is
only ever an eval set, never trained on" premise (so no train/eval molecule
pool split is needed here either).

Usage (from paper/, on a GPU allocation):
    python dev/eval_qm9_checkpoint.py --targets all --n_graphs_min 100 --n_graphs_max 300
    python dev/eval_qm9_checkpoint.py --targets homo,lumo,gap --n_graphs_min 500 --n_graphs_max 1000

Notes on context size: QM9 molecules are small (real organics, ~9-29 atoms
including hydrogens) compared to the synthetic prior's base_n_nodes (15-60),
so even n_graphs=1000 here (~9k-29k total atoms) stays well under the
~120,000-atom point that OOM'd a 40GB GPU (see graph_level_pooling.md, section
11) -- but it's also well beyond the 100-300 range the pooler was actually
trained on, so treat very large contexts as an extrapolation test, not
guaranteed to help.
"""

import argparse
import sys
import tomllib
from pathlib import Path

REPO = Path("/p/project1/profound/al-zeqri1/PFN/second/graphpfn/paper")
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score

import lib.tfm.limix as limix_mod

_LOCAL_CKPT = str(REPO / "checkpoints/LimiX-16M.ckpt")
limix_mod._download_limix_checkpoint = lambda: _LOCAL_CKPT

from lib.graphpfn.model import GraphPFN
from lib.graphpfn.pooling import GraphLevelGraphPFN
from lib.graphpfn.qm9_data import QM9_TARGETS, load_qm9, sample_one_qm9_dataset

DEFAULT_OUTPUT_DIR = (
    REPO / "exp/graphpfn/pretrain/multigraph_molecule_graph_level_pooling/pretrain"
)
DEFAULT_QM9_ROOT = "/p/project1/profound/al-zeqri1/PFN/graphpfn/data/QM9"

# The standard 12-target QM9 benchmark subset used across the equivariant-GNN
# literature (MPNN/SchNet/Cormorant/L1Net/LieConv/DimeNet++/TFN/SE(3)-Tr./EGNN
# -- e.g. the E(n)-GNN paper's Table 3): alpha, Delta-epsilon (=gap),
# eps_HOMO, eps_LUMO, mu, Cv, G, H, R^2 (=r2), U, U0, ZPVE.
#
# U/U0/H/G here map to the *atomization-energy* columns (u298_atom/u0_atom/
# h298_atom/g298_atom), NOT the raw total-energy columns (u298/u0/h298/g298):
# since Gilmer et al. 2017, this literature reports U/U0/H/G as energy
# relative to isolated-atom references, not raw total electronic energy --
# the raw columns include every atom's core-electron energy (thousands of eV,
# scaling with composition), which would make MAE numbers look wildly
# different from anything in that table for reasons having nothing to do
# with model quality. Rotational constants A/B/C don't appear in that table.
CORE12_TARGETS = [
    "alpha", "gap", "homo", "lumo", "mu", "cv",
    "g298_atom", "h298_atom", "r2", "u298_atom", "u0_atom", "zpve",
]


def context_query_split(n_graphs: int, train_ratio: float, rng: np.random.Generator):
    n_train = max(1, min(n_graphs - 1, round(n_graphs * train_ratio)))
    perm = rng.permutation(n_graphs)
    return perm[:n_train], perm[n_train:]


def standardize_by_context(y_all: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, float]:
    """Returns (y_standardized, std) -- `std` is also returned so callers can
    rescale a standardized-scale MAE back to the target's native units
    (MAE is scale-equivariant: MAE(std*y) == std*MAE(y))."""
    mean = y_all.mean()
    std = y_all.std()
    std = std if std > 1e-6 else 1.0
    return (y_all - mean) / std, std


def build_model(output_dir: Path, device: torch.device) -> GraphLevelGraphPFN:
    """Rebuilds the exact architecture pretrain_graph_level.py trained (model
    kwargs from the run's own pretrain.toml, sitting one directory above the
    output dir by convention), then loads the EMA weights from checkpoint.pt
    -- eval_fn in pretrain_graph_level.py always evaluates model_ema, not the
    raw (non-averaged) weights, so this matches the numbers that script itself
    reported during training.
    """
    toml_path = output_dir.parent / "pretrain.toml"
    with open(toml_path, "rb") as f:
        config = tomllib.load(f)["base_config"]

    model_kwargs = dict(config.get("model", {}))
    model_kwargs.setdefault("edge_head", None)
    graphpfn = GraphPFN(**model_kwargs)
    model = GraphLevelGraphPFN(
        graphpfn, embed_dim=graphpfn.tfm.module.embed_dim, **config.get("pooling", {})
    )

    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    print(f"loaded checkpoint from step={checkpoint['step']}")
    ema_state = {
        k[len("module."):]: v
        for k, v in checkpoint["model_ema"].items()
        if k.startswith("module.")
    }
    model.load_state_dict(ema_state)

    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def eval_one_target(
    model: GraphLevelGraphPFN,
    dataset,
    target: str,
    target_idx: int,
    n_graphs_range: tuple[int, int],
    train_ratio: float,
    n_eval_datasets: int,
    device: torch.device,
    rng: np.random.Generator,
    max_attempts: int = 50,
) -> dict[str, np.ndarray]:
    r2_scores = []
    mae_std_scores = []
    mae_raw_scores = []
    for _ in range(n_eval_datasets):
        for _ in range(max_attempts):
            atom_features, graph, y_per_molecule = sample_one_qm9_dataset(
                dataset, n_graphs_range, target_idx, rng
            )
            n_graphs = int(graph.batch_num_nodes().shape[0])
            train_idx, test_idx = context_query_split(n_graphs, train_ratio, rng)
            y_np = y_per_molecule.numpy()
            if len(test_idx) > 0 and len(train_idx) >= 2 and np.std(y_np[test_idx]) != 0:
                break
        else:
            raise RuntimeError(f"could not sample a valid split for target={target!r}")

        y_std, y_scale = standardize_by_context(y_np, train_idx)
        is_context = torch.zeros(n_graphs, dtype=torch.bool, device=device)
        is_context[torch.from_numpy(train_idx)] = True
        y_std_t = torch.as_tensor(y_std, dtype=torch.float32, device=device)

        pred = model(
            graph.to(device), atom_features.to(device), y_std_t, is_context
        )
        query_idx = torch.from_numpy(test_idx).to(device)
        y_true_np = y_std_t[query_idx].cpu().numpy()
        y_pred_np = pred[query_idx].float().cpu().numpy()

        r2_scores.append(r2_score(y_true_np, y_pred_np))
        mae_std = mean_absolute_error(y_true_np, y_pred_np)
        mae_std_scores.append(mae_std)
        mae_raw_scores.append(mae_std * y_scale)  # rescaled to the target's native units

    return {
        "r2": np.array(r2_scores),
        "mae_std": np.array(mae_std_scores),
        "mae_raw": np.array(mae_raw_scores),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--targets", type=str, default="core12",
        help=(
            "Comma-separated QM9 target names, 'all' for every one of the 19 standard "
            "targets, or 'core12' for the standard 12-target equivariant-GNN benchmark "
            "subset (alpha, gap, homo, lumo, mu, cv, g298, h298, r2, u298, u0, zpve)."
        ),
    )
    parser.add_argument("--n_graphs_min", type=int, default=100)
    parser.add_argument("--n_graphs_max", type=int, default=300)
    parser.add_argument("--train_ratio", type=float, default=0.3)
    parser.add_argument("--n_eval_datasets", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--qm9_root", type=str, default=DEFAULT_QM9_ROOT)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "This script needs a real GPU allocation."
    device = torch.device("cuda")

    if args.targets == "all":
        targets = QM9_TARGETS
    elif args.targets == "core12":
        targets = CORE12_TARGETS
    else:
        targets = [t.strip() for t in args.targets.split(",")]
    for t in targets:
        assert t in QM9_TARGETS, f"unknown QM9 target {t!r}, must be one of {QM9_TARGETS}"

    print(f"building model from {args.output_dir} ...")
    model = build_model(args.output_dir, device)

    print(f"loading QM9 from {args.qm9_root} ...")
    dataset = load_qm9(args.qm9_root)
    print(f"loaded QM9: {len(dataset)} molecules")

    rng = np.random.default_rng(args.seed)
    n_graphs_range = (args.n_graphs_min, args.n_graphs_max)
    print(
        f"context size: {n_graphs_range[0]}-{n_graphs_range[1]} molecules, "
        f"train_ratio={args.train_ratio}, n_eval_datasets={args.n_eval_datasets}"
    )

    results = {}
    for target in targets:
        target_idx = QM9_TARGETS.index(target)
        metrics = eval_one_target(
            model, dataset, target, target_idx, n_graphs_range, args.train_ratio,
            args.n_eval_datasets, device, rng,
        )
        results[target] = metrics
        r2, mae_raw, mae_std = metrics["r2"], metrics["mae_raw"], metrics["mae_std"]
        print(
            f"{target:>10s}: mean R^2={r2.mean():+.3f}  median R^2={np.median(r2):+.3f}  "
            f"MAE={mae_raw.mean():.4g} (native units)  MAE_std={mae_std.mean():.3f}  "
            f"(n={len(r2)})"
        )

    print("\n=== summary (sorted by R^2) ===")
    for target, metrics in sorted(results.items(), key=lambda kv: -kv[1]["r2"].mean()):
        print(
            f"{target:>10s}: R^2={metrics['r2'].mean():+.3f}  "
            f"MAE={metrics['mae_raw'].mean():.4g}"
        )


if __name__ == "__main__":
    main()
