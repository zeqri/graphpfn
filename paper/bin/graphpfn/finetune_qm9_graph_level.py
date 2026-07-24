"""Finetune GraphLevelGraphPFN on a single real QM9 regression target via
gradient-based ICL training on real molecules -- unlike
dev/eval_qm9_checkpoint.py (zero-shot, no gradient) or
pretrain_graph_level.py's periodic QM9 eval (also zero-shot, QM9 is never
trained on there), this actually backpropagates through real QM9 labels.

Starts from a pretrained checkpoint (bin/graphpfn/pretrain_graph_level.py's
output) and continues training on real QM9 data instead of the synthetic
prior, using the exact same GraphLevelGraphPFN.forward /
context_query_split / standardize_by_context conventions used everywhere
else in this pipeline.

QM9 molecules are split into a train pool and a held-out test pool by
molecule index (seeded, once, up front): training only ever samples
context/query molecules from the train pool, evaluation only ever samples
from the test pool -- so reported R^2/MAE reflect genuine generalization to
molecules never seen during finetuning, not molecules the model has already
been shown (as context or query) many times over.

Usage (from paper/, on a GPU allocation):
    python bin/graphpfn/finetune_qm9_graph_level.py --target gap
    python bin/graphpfn/finetune_qm9_graph_level.py --target mu --unfreeze_backbone --n_steps 2000

Results (R^2, MAE in the target's native units, and standardized-scale MAE)
are printed at every eval and written to
exp/graphpfn/finetune/qm9_<target>/{training_log.jsonl,results.json}, plus
model checkpoints (checkpoint_last.pt, checkpoint_best.pt by held-out R^2).
"""

import argparse
import json
import sys
import tomllib
from pathlib import Path

REPO = Path("/p/project1/profound/al-zeqri1/PFN/second/graphpfn/paper")
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score

import lib.deep
import lib.tfm.limix as limix_mod

_LOCAL_CKPT = str(REPO / "checkpoints/LimiX-16M.ckpt")
limix_mod._download_limix_checkpoint = lambda: _LOCAL_CKPT

from lib.graphpfn.model import GraphPFN
from lib.graphpfn.pooling import GraphLevelGraphPFN
from lib.graphpfn.qm9_data import QM9_TARGETS, load_qm9, sample_one_qm9_dataset

DEFAULT_PRETRAIN_DIR = (
    REPO / "exp/graphpfn/pretrain/multigraph_molecule_graph_level_pooling/pretrain"
)
DEFAULT_QM9_ROOT = "/p/project1/profound/al-zeqri1/PFN/graphpfn/data/QM9"
DEFAULT_FINETUNE_ROOT = REPO / "exp/graphpfn/finetune"


def context_query_split(n_graphs: int, train_ratio: float, rng: np.random.Generator):
    n_train = max(1, min(n_graphs - 1, round(n_graphs * train_ratio)))
    perm = rng.permutation(n_graphs)
    return perm[:n_train], perm[n_train:]


def standardize_by_context(y_all: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, float]:
    """Returns (y_standardized, std) -- `std` also returned so callers can
    rescale a standardized-scale MAE back to the target's native units."""
    mean = y_all.mean()
    std = y_all.std()
    std = std if std > 1e-6 else 1.0
    return (y_all - mean) / std, std


def build_model(
    pretrain_dir: Path,
    unfreeze_backbone: bool,
    device: torch.device,
    skip_pretrained: bool = False,
    pooler_type: str | None = None,
) -> GraphLevelGraphPFN:
    """Same convention as dev/eval_qm9_checkpoint.py's build_model: rebuild
    the exact architecture pretrain_graph_level.py trained (model/pooling
    kwargs from the run's own pretrain.toml, sitting one directory above the
    output dir by convention), then load the EMA weights from checkpoint.pt.

    `skip_pretrained=True` is the "untrained checkpoint" ablation: keeps the
    LimiX backbone's own tabular pretraining (loaded automatically inside
    GraphPFN.__init__, unrelated to this project) but skips loading
    checkpoint.pt's EMA weights for the 12-layer graph adapter (zero-init by
    default, so a true no-op at this point) and the geometric_attention/
    pooler modules (real, non-zero random init) -- isolating whether
    pretrain_graph_level.py's synthetic-prior pretraining stage contributes
    anything beyond what direct QM9 finetuning achieves on its own. Mirrors
    the parent paper's own "random graph adapters" ablation (Appendix C,
    Table 7's "LimiX+GA (FT)" row) for the node-level pipeline.

    `pooler_type` overrides the pretrain.toml's own pooling config (e.g.
    "sum" for the EGNN-style SumPooler instead of the default attention
    pooler, see lib/graphpfn/pooling.py). Only valid together with
    skip_pretrained=True -- a non-default pooler's parameters won't match
    checkpoint.pt's saved state dict, so loading it would fail (or silently
    load the wrong module if names happened to collide).
    """
    if pooler_type is not None and pooler_type != "attention" and not skip_pretrained:
        raise ValueError(
            f"pooler_type={pooler_type!r} requires skip_pretrained=True -- the pretrained "
            "checkpoint's pooler weights were saved for the default 'attention' pooler and "
            "won't match a different pooler architecture's state dict."
        )

    toml_path = pretrain_dir.parent / "pretrain.toml"
    with open(toml_path, "rb") as f:
        config = tomllib.load(f)["base_config"]

    model_kwargs = dict(config.get("model", {}))
    model_kwargs.setdefault("edge_head", None)
    if unfreeze_backbone:
        # Full-finetune regime (paper convention: "finetune the entire
        # model"), as opposed to the ICL/pretraining regime the backbone was
        # trained under (freeze_tfm=True, only the adapter/pooling trains).
        model_kwargs["freeze_tfm"] = False
    graphpfn = GraphPFN(**model_kwargs)
    pooling_kwargs = dict(config.get("pooling", {}))
    if pooler_type is not None:
        pooling_kwargs["pooler_type"] = pooler_type
    model = GraphLevelGraphPFN(graphpfn, embed_dim=graphpfn.tfm.module.embed_dim, **pooling_kwargs)

    if skip_pretrained:
        print(
            "skip_pretrained=True: keeping LimiX's own tabular pretraining, but leaving the "
            "graph adapter / geometric_attention / pooler at their fresh initialization "
            "(NOT loading pretrain_graph_level.py's checkpoint.pt)"
        )
        return model.to(device)

    checkpoint = torch.load(pretrain_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    print(f"loaded pretrained checkpoint from step={checkpoint['step']}")
    ema_state = {
        k[len("module."):]: v
        for k, v in checkpoint["model_ema"].items()
        if k.startswith("module.")
    }
    model.load_state_dict(ema_state)
    return model.to(device)


def sample_valid_split(
    dataset,
    n_graphs_range: tuple[int, int],
    target_idx: int,
    train_ratio: float,
    rng: np.random.Generator,
    max_attempts: int = 50,
    all_pairs: bool = False,
):
    for _ in range(max_attempts):
        atom_features, graph, y_per_molecule = sample_one_qm9_dataset(
            dataset, n_graphs_range, target_idx, rng, all_pairs=all_pairs
        )
        n_graphs = int(graph.batch_num_nodes().shape[0])
        train_idx, test_idx = context_query_split(n_graphs, train_ratio, rng)
        y_np = y_per_molecule.numpy()
        if len(test_idx) > 0 and len(train_idx) >= 2 and np.std(y_np[test_idx]) != 0:
            return atom_features, graph, y_np, train_idx, test_idx, n_graphs
    raise RuntimeError("could not sample a valid context/query split after max_attempts")


def forward_one_batch(
    model: GraphLevelGraphPFN,
    device: torch.device,
    amp_enabled: bool,
    atom_features: torch.Tensor,
    graph,
    y_np: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
):
    n_graphs = int(graph.batch_num_nodes().shape[0])
    y_std, y_scale = standardize_by_context(y_np, train_idx)
    is_context = torch.zeros(n_graphs, dtype=torch.bool, device=device)
    is_context[torch.from_numpy(train_idx)] = True
    y_std_t = torch.as_tensor(y_std, dtype=torch.float32, device=device)

    with torch.autocast(
        device.type, enabled=amp_enabled, dtype=torch.bfloat16 if amp_enabled else None
    ):
        pred = model(graph.to(device), atom_features.to(device), y_std_t, is_context)

    query_idx = torch.from_numpy(test_idx).to(device)
    return pred[query_idx], y_std_t[query_idx], y_scale


@torch.no_grad()
def evaluate(
    model: GraphLevelGraphPFN,
    test_dataset,
    target_idx: int,
    n_graphs_range: tuple[int, int],
    train_ratio: float,
    n_eval_datasets: int,
    device: torch.device,
    rng: np.random.Generator,
    all_pairs: bool = False,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    r2_scores, mae_std_scores, mae_raw_scores = [], [], []
    for _ in range(n_eval_datasets):
        atom_features, graph, y_np, train_idx, test_idx, _ = sample_valid_split(
            test_dataset, n_graphs_range, target_idx, train_ratio, rng, all_pairs=all_pairs
        )
        pred, target, y_scale = forward_one_batch(
            model, device, False, atom_features, graph, y_np, train_idx, test_idx
        )
        y_true_np = target.cpu().numpy()
        y_pred_np = pred.float().cpu().numpy()
        r2_scores.append(r2_score(y_true_np, y_pred_np))
        mae_std = mean_absolute_error(y_true_np, y_pred_np)
        mae_std_scores.append(mae_std)
        mae_raw_scores.append(mae_std * y_scale)
    if was_training:
        model.train()
    return {
        "r2": float(np.mean(r2_scores)),
        "mae": float(np.mean(mae_raw_scores)),
        "mae_std": float(np.mean(mae_std_scores)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True, choices=QM9_TARGETS)
    parser.add_argument("--pretrain_dir", type=Path, default=DEFAULT_PRETRAIN_DIR)
    parser.add_argument(
        "--unfreeze_backbone", action="store_true",
        help="Full finetune (unfreeze the LimiX backbone too), not just the adapter/pooling.",
    )
    parser.add_argument(
        "--skip_pretrained", action="store_true",
        help=(
            "Ablation: skip loading pretrain_graph_level.py's checkpoint.pt (graph adapter / "
            "geometric_attention / pooler start from fresh initialization instead of the "
            "synthetic-prior-pretrained weights). LimiX's own tabular pretraining still loads "
            "as usual. Use this to test whether the graph-level pretraining stage actually "
            "helps, vs. finetuning directly on QM9 from an otherwise-fresh model."
        ),
    )
    parser.add_argument(
        "--all_pairs_distance", action="store_true",
        help=(
            "Use a fully-connected per-molecule graph (every atom pair, real 3D-coordinate "
            "distances) for geometric attention instead of only chemical-bond edges -- tests "
            "whether restricting to 1-hop bonded distances (vs. EGNN's own QM9 setup, which "
            "uses all pairs) is what caps whole-molecule-shape-dependent targets like A/B/C/r2. "
            "No model changes needed -- GeometricAttentionStack just attends over whatever "
            "edges the input graph has."
        ),
    )
    parser.add_argument(
        "--pooler_type", type=str, default=None, choices=["attention", "sum"],
        help=(
            "Override the pooling mechanism. 'sum' is the EGNN-style readout (per-atom MLP, "
            "summed over each molecule's atoms, then a post-pooling MLP) instead of the "
            "default learned-attention pooler -- better suited to extensive properties (total "
            "energy, zpve) that scale with atom count. Requires --skip_pretrained (a non-"
            "default pooler won't match the pretrained checkpoint's saved weights)."
        ),
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--qm9_root", type=str, default=DEFAULT_QM9_ROOT)
    parser.add_argument(
        "--test_fraction", type=float, default=0.1,
        help="Fraction of QM9 molecules held out for eval, never sampled during training.",
    )
    parser.add_argument("--n_graphs_min", type=int, default=100)
    parser.add_argument("--n_graphs_max", type=int, default=300)
    parser.add_argument("--train_ratio", type=float, default=0.3)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--n_eval_datasets", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--n_warmup_steps", type=int, default=50)
    parser.add_argument("--gradient_clipping_norm", type=float, default=1.0)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = (not args.no_amp) and device.type == "cuda"
    print(f"device={device}, amp_enabled={amp_enabled}")

    default_name = (
        f"qm9_{args.target}"
        + ("_scratch" if args.skip_pretrained else "")
        + ("_allpairs" if args.all_pairs_distance else "")
        + (f"_{args.pooler_type}pooler" if args.pooler_type not in (None, "attention") else "")
    )
    output_dir = args.output_dir or (DEFAULT_FINETUNE_ROOT / default_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir = {output_dir}")

    target_idx = QM9_TARGETS.index(args.target)
    rng = np.random.default_rng(args.seed)
    n_graphs_range = (args.n_graphs_min, args.n_graphs_max)

    print(f"loading QM9 from {args.qm9_root} ...")
    dataset = load_qm9(args.qm9_root)
    n_molecules = len(dataset)
    perm = rng.permutation(n_molecules)
    n_test = max(1, round(n_molecules * args.test_fraction))
    test_indices = perm[:n_test]
    train_indices = perm[n_test:]
    train_dataset = dataset[torch.as_tensor(train_indices)]
    test_dataset = dataset[torch.as_tensor(test_indices)]
    print(
        f"QM9: {n_molecules} molecules -> {len(train_dataset)} train / "
        f"{len(test_dataset)} held-out test (never sampled during training)"
    )

    print(
        f"building model from {args.pretrain_dir} "
        f"(unfreeze_backbone={args.unfreeze_backbone}, skip_pretrained={args.skip_pretrained}, "
        f"pooler_type={args.pooler_type}) ..."
    )
    model = build_model(
        args.pretrain_dir, args.unfreeze_backbone, device,
        skip_pretrained=args.skip_pretrained, pooler_type=args.pooler_type,
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_trainable}/{n_total}")

    # make_parameter_groups doesn't filter by requires_grad (frozen params are
    # harmless to pass to the optimizer -- they never accumulate a gradient,
    # so AdamW just skips them), matching pretrain_graph_level.py's own
    # convention exactly.
    params = lib.deep.make_parameter_groups(model)
    optimizer = lib.deep.make_optimizer(
        type="AdamW", lr=args.lr, weight_decay=args.weight_decay, params=params
    )
    lr_scheduler = lib.deep.get_lr_scheduler(
        optimizer=optimizer, n_warmup_steps=args.n_warmup_steps, n_steps=args.n_steps,
        scheduler="cosine",
    )

    log_path = output_dir / "training_log.jsonl"

    def log_and_print(step: int, metrics: dict, extra: dict | None = None) -> None:
        entry = {"step": step, **metrics, **(extra or {}), "lr": lib.deep.get_lr(optimizer)}
        print(
            f"[step {step:5d}] target={args.target}: R^2={metrics['r2']:+.4f}  "
            f"MAE={metrics['mae']:.4g} (native units)  MAE_std={metrics['mae_std']:.4f}"
        )
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    print("\n=== baseline (before finetuning) ===")
    best_r2 = -float("inf")
    metrics = evaluate(
        model, test_dataset, target_idx, n_graphs_range, args.train_ratio,
        args.n_eval_datasets, device, rng, all_pairs=args.all_pairs_distance,
    )
    log_and_print(0, metrics)
    best_r2 = metrics["r2"]
    torch.save(model.state_dict(), output_dir / "checkpoint_best.pt")

    model.train()
    print(f"\n=== finetuning on target={args.target!r} for {args.n_steps} steps ===")
    for step in range(1, args.n_steps + 1):
        atom_features, graph, y_np, train_idx, test_idx, _ = sample_valid_split(
            train_dataset, n_graphs_range, target_idx, args.train_ratio, rng,
            all_pairs=args.all_pairs_distance,
        )
        pred, target, _ = forward_one_batch(
            model, device, amp_enabled, atom_features, graph, y_np, train_idx, test_idx
        )
        loss = F.mse_loss(pred, target)

        if not torch.isfinite(loss):
            print(f"[step {step}] non-finite loss ({loss.item()}), skipping step")
            continue

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping_norm)
        if not torch.isfinite(grad_norm):
            print(f"[step {step}] non-finite grad norm, skipping step")
            optimizer.zero_grad()
            continue
        optimizer.step()
        lr_scheduler.step()

        if step % args.eval_every == 0 or step == args.n_steps:
            metrics = evaluate(
                model, test_dataset, target_idx, n_graphs_range, args.train_ratio,
                args.n_eval_datasets, device, rng, all_pairs=args.all_pairs_distance,
            )
            log_and_print(step, metrics, extra={"loss": loss.item()})
            torch.save(model.state_dict(), output_dir / "checkpoint_last.pt")
            if metrics["r2"] > best_r2:
                best_r2 = metrics["r2"]
                torch.save(model.state_dict(), output_dir / "checkpoint_best.pt")

    print("\n=== final results ===")
    final_metrics = evaluate(
        model, test_dataset, target_idx, n_graphs_range, args.train_ratio,
        args.n_eval_datasets, device, rng, all_pairs=args.all_pairs_distance,
    )
    print(
        f"target={args.target}: R^2={final_metrics['r2']:+.4f}  "
        f"MAE={final_metrics['mae']:.4g} (native units)  "
        f"MAE_std={final_metrics['mae_std']:.4f}  best_R^2_seen={best_r2:+.4f}"
    )
    with open(output_dir / "results.json", "w") as f:
        json.dump(
            {
                "target": args.target,
                "final": final_metrics,
                "best_r2": best_r2,
                "n_steps": args.n_steps,
                "unfreeze_backbone": args.unfreeze_backbone,
                "all_pairs_distance": args.all_pairs_distance,
                "n_graphs_range": list(n_graphs_range),
                "train_ratio": args.train_ratio,
                "test_fraction": args.test_fraction,
            },
            f,
            indent=2,
        )
    print(f"results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
