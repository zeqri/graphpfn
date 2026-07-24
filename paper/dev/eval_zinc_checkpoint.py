"""Evaluate a trained GraphLevelGraphPFN checkpoint (from
bin/graphpfn/pretrain_graph_level.py) via zero-gradient ICL on real ZINC
molecules -- the ZINC counterpart to dev/eval_qm9_checkpoint.py, sharing all
of its conventions (context/query split, standardize_by_context, "real data
is only ever an eval set, never trained on") except that ZINC has a single
regression target (constrained solubility, `y`) instead of QM9's 19 columns,
so there is no --targets/target_idx machinery here.

Usage (from paper/, on a GPU allocation):
    python dev/eval_zinc_checkpoint.py --n_graphs_min 100 --n_graphs_max 300

Notes on context size: see eval_qm9_checkpoint.py's docstring -- the same
extrapolation caveat applies here (n_graphs beyond a few hundred goes well
past what the pooler was trained on).
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
from lib.graphpfn.zinc_data import load_zinc, sample_one_zinc_dataset

DEFAULT_OUTPUT_DIR = (
    REPO / "exp/graphpfn/pretrain/multigraph_molecule_graph_level_pooling/pretrain"
)
DEFAULT_ZINC_ROOT = "/p/project1/profound/al-zeqri1/PFN/graphpfn/data/ZINC"


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
def eval_zinc(
    model: GraphLevelGraphPFN,
    dataset,
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
            atom_features, graph, y_per_molecule = sample_one_zinc_dataset(
                dataset, n_graphs_range, rng
            )
            n_graphs = int(graph.batch_num_nodes().shape[0])
            train_idx, test_idx = context_query_split(n_graphs, train_ratio, rng)
            y_np = y_per_molecule.numpy()
            if len(test_idx) > 0 and len(train_idx) >= 2 and np.std(y_np[test_idx]) != 0:
                break
        else:
            raise RuntimeError("could not sample a valid split for ZINC")

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
    parser.add_argument("--n_graphs_min", type=int, default=100)
    parser.add_argument("--n_graphs_max", type=int, default=300)
    parser.add_argument("--train_ratio", type=float, default=0.3)
    parser.add_argument("--n_eval_datasets", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--zinc_root", type=str, default=DEFAULT_ZINC_ROOT)
    parser.add_argument(
        "--full", action="store_true",
        help="Use the full ~250k-molecule ZINC dataset instead of the standard 12k subset.",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "This script needs a real GPU allocation."
    device = torch.device("cuda")

    print(f"building model from {args.output_dir} ...")
    model = build_model(args.output_dir, device)

    print(f"loading ZINC from {args.zinc_root} (subset={not args.full}) ...")
    dataset = load_zinc(args.zinc_root, subset=not args.full)
    print(f"loaded ZINC: {len(dataset)} molecules")

    rng = np.random.default_rng(args.seed)
    n_graphs_range = (args.n_graphs_min, args.n_graphs_max)
    print(
        f"context size: {n_graphs_range[0]}-{n_graphs_range[1]} molecules, "
        f"train_ratio={args.train_ratio}, n_eval_datasets={args.n_eval_datasets}"
    )

    metrics = eval_zinc(
        model, dataset, n_graphs_range, args.train_ratio, args.n_eval_datasets, device, rng,
    )
    r2, mae_raw, mae_std = metrics["r2"], metrics["mae_raw"], metrics["mae_std"]
    print(
        f"      zinc: mean R^2={r2.mean():+.3f}  median R^2={np.median(r2):+.3f}  "
        f"MAE={mae_raw.mean():.4g} (native units)  MAE_std={mae_std.mean():.3f}  "
        f"(n={len(r2)})"
    )


if __name__ == "__main__":
    main()
