"""Same diagnostic as dev/diagnose_conv_type_size_r2.py, but for checkpoints
trained by
limix_backbone_gnn_pooling_multi_dataset_multi_view_pooler_fit_test.py (the
MultiViewPool architecture, which stops collapsing mean/min/max/attention
pooling into one embedding and instead emits them as separate groups for the
frozen backbone's own cross-group attention to select among -- see that
script's module docstring). The two checkpoints are NOT interchangeable:
this file's PoolingGNN/MultiViewPool must match
limix_backbone_gnn_pooling_multi_dataset_multi_view_pooler_fit_test.py
exactly (state_dict shapes differ from the multi_agg_pooler/MultiAggregatorPool
version diagnose_conv_type_size_r2.py loads).

Loads an already-trained PoolingGNN checkpoint (pooler_ema weights) and runs
it, read-only, on many freshly sampled held-out datasets PER conv_type, with
conv_type FORCED (not left to chance) and molecule count spread across the
full [--min-molecules, --max-molecules] range -- giving, for each conv_type
separately, a real R2-vs-n_molecules trend instead of the training script's
own small (N_EVAL_DATASETS=8), by-chance-split held-out set.

All held-out datasets here use a seed range (--seed-base, default 1_000_000)
disjoint from both the training draws and the training script's own tracked
eval set (EVAL_SEED_BASE=10_000) -- this is a separate, one-off analysis, not
part of the tracked training metric.

Usage:
    python dev/diagnose_conv_type_size_r2_multi_view_pooler.py \\
        --checkpoint-path dev/output/limix_backbone_gnn_pooling_multi_dataset_multi_view_pooler_fit_test/molecules_1000_3000_n_steps_10000_lr_0.003/pooler_checkpoint.pt \\
        --min-molecules 500 --max-molecules 4000 --n-samples-per-conv-type 25 --n-workers 8
(Single-process, single-GPU-or-CPU -- no torchrun needed, this is read-only
inference over a fixed checkpoint, not a training job.)
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent.parent
if str(PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_DIR))

import dgl  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from lib.graphpfn.prior.attributes import sample_graph_level_labels_via_virtual_node  # noqa: E402
from lib.graphpfn.prior.config import sample_configs  # noqa: E402
from lib.graphpfn.prior.graphs.multi_graph import sample_multi_graph  # noqa: E402
from lib.graphpfn.prior.postprocessing import process_features  # noqa: E402
from lib.graphpfn.prior.prior_typings import unpack  # noqa: E402
from vendor.limix.utils.loading import load_model  # noqa: E402
from dev.limix_encoder_pooling_probe import BASE_PRIOR_CONFIG  # noqa: E402

CHECKPOINT_PATH = PAPER_DIR / "checkpoints" / "LimiX-16M.ckpt"
TRAIN_FRACTION = 0.8

CONV_TYPES = ["gcn", "sage-mean", "sage-min", "sage-max", "gt"]

# Must match the architecture the checkpoint was actually trained with (see
# limix_backbone_gnn_pooling_multi_dataset_multi_view_pooler_fit_test.py) --
# these aren't guessed, loading will fail loudly (state_dict shape mismatch)
# if they're wrong.
N_GNN_LAYERS = 3
N_ATTN_HEADS = 4
N_POOL_VIEWS = 4  # mean, min, max, learned-attention -- see MultiViewPool
DROPOUT = 0.0
EMA_DECAY = 0.98  # unused for inference (never updated further), just needed to construct AveragedModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Trained pooler_checkpoint.pt to diagnose.")
    parser.add_argument("--min-molecules", type=int, default=500, help="Lower bound for molecules/dataset (default: 500).")
    parser.add_argument("--max-molecules", type=int, default=4000, help="Upper bound for molecules/dataset (default: 4000).")
    parser.add_argument(
        "--n-samples-per-conv-type",
        type=int,
        default=25,
        help="Held-out datasets to sample PER conv_type (default: 25) -- total samples = 5x this.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=8,
        help="Parallel CPU worker processes for sampling (default: 8) -- this is a one-shot batch, "
        "not a continuous training loop, so a plain multiprocessing.Pool is used instead of the "
        "training scripts' background-prefetch DataLoader.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=1_000_000,
        help="Base seed for sampled datasets (default: 1_000_000) -- kept disjoint from training "
        "draws and from the training script's own EVAL_SEED_BASE=10_000 tracked eval set, since this "
        "is a separate one-off analysis.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Where to save raw per-sample (conv_type, n_molecules, r2) records as JSON. Defaults to "
        "'conv_type_size_diagnostic.json' next to --checkpoint-path.",
    )
    return parser.parse_args()


def build_prior_config_for_conv_type(min_molecules: int, max_molecules: int, conv_type: str) -> dict:
    """Like the training script's build_prior_config, but FORCES conv_type
    to a single fixed value instead of leaving it a random choice -- this
    script needs conv_type controlled, not sampled, so every conv_type gets
    a real, deliberate spread of molecule sizes instead of whatever a random
    draw happens to produce.
    """
    prior_config = copy.deepcopy(BASE_PRIOR_CONFIG)
    prior_values = prior_config["prior"]["values"][0]

    n_graphs_dist = {"_distribution_": "log_uniform_int", "min": min_molecules, "max": max_molecules}
    prior_values["graph"]["sampler"]["values"][0]["n_graphs"] = n_graphs_dist
    prior_values["graph"]["n_nodes"] = n_graphs_dist  # cosmetic only, see the training script's own note

    prior_values["scm"]["conv_type"] = {"_distribution_": "choice", "values": [conv_type]}

    return prior_config


class MultiAggregatorConv(nn.Module):
    """Per-layer message passing combining mean/min/max neighbor reductions
    with a lightweight multi-head dot-product attention aggregation,
    concatenated and projected back to d_out. Copied verbatim from
    limix_backbone_gnn_pooling_multi_dataset_multi_view_pooler_fit_test.py --
    must match exactly for the checkpoint's state_dict to load. Unlike the
    readout (MultiViewPool below), this per-layer step still combines
    branches -- see that script's module docstring for why.
    """

    def __init__(self, d_in: int, d_out: int, n_heads: int = 1):
        super().__init__()
        assert d_out % n_heads == 0, f"d_out={d_out} must be divisible by n_heads={n_heads}"
        self.d_out = d_out
        self.n_heads = n_heads
        self.d_head = d_out // n_heads
        self.attn_scale = self.d_head**-0.5

        self.pre_linear = nn.Linear(d_in, d_out)
        self.attn_qkv = nn.Linear(d_out, d_out * 3)
        self.attn_out = nn.Linear(d_out, d_out)
        self.combine = nn.Linear(d_out * 5, d_out)

    def forward(self, graph: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        h = self.pre_linear(x)

        mean_msg = dgl.ops.copy_u_mean(graph, h)
        min_msg = dgl.ops.copy_u_min(graph, h)
        max_msg = dgl.ops.copy_u_max(graph, h)

        graph_sl = dgl.add_self_loop(graph)
        qkv = self.attn_qkv(h).reshape(-1, self.n_heads, self.d_head * 3)
        q, k, v = qkv.split((self.d_head, self.d_head, self.d_head), dim=-1)
        attn_scores = dgl.ops.u_dot_v(graph_sl, k, q) * self.attn_scale
        attn_probs = dgl.ops.edge_softmax(graph_sl, attn_scores)
        attn_msg = dgl.ops.u_mul_e_sum(graph_sl, v, attn_probs).reshape(-1, self.d_out)
        attn_msg = self.attn_out(attn_msg)

        combined = torch.cat([h, mean_msg, min_msg, max_msg, attn_msg], dim=-1)
        return self.combine(combined)


def _scatter_pool(atom_out: torch.Tensor, molecule_id: torch.Tensor, n_molecules: int, reduce: str, init_value: float) -> torch.Tensor:
    n_atoms, embed_dim = atom_out.shape
    out = torch.full((n_molecules, embed_dim), init_value, device=atom_out.device, dtype=atom_out.dtype)
    index = molecule_id.unsqueeze(-1).expand(-1, embed_dim)
    return out.scatter_reduce(0, index, atom_out, reduce=reduce, include_self=False)


class MultiViewPool(nn.Module):
    """Atom->molecule readout emitting mean/min/max/attention pooling as
    SEPARATE groups instead of collapsing them into one embedding. Copied
    verbatim from
    limix_backbone_gnn_pooling_multi_dataset_multi_view_pooler_fit_test.py --
    must match exactly for the checkpoint's state_dict to load.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.attn_score = nn.Linear(embed_dim, 1)

    def forward(self, atom_out: torch.Tensor, molecule_id: torch.Tensor, n_molecules: int) -> torch.Tensor:
        mean_pool = _scatter_pool(atom_out, molecule_id, n_molecules, reduce="mean", init_value=0.0)
        min_pool = _scatter_pool(atom_out, molecule_id, n_molecules, reduce="amin", init_value=float("inf"))
        max_pool = _scatter_pool(atom_out, molecule_id, n_molecules, reduce="amax", init_value=float("-inf"))
        attn_pool = self._attention_pool(atom_out, molecule_id, n_molecules)
        return torch.stack([mean_pool, min_pool, max_pool, attn_pool], dim=1)  # (n_molecules, N_POOL_VIEWS, embed_dim)

    def _attention_pool(self, atom_out: torch.Tensor, molecule_id: torch.Tensor, n_molecules: int) -> torch.Tensor:
        device, dtype = atom_out.device, atom_out.dtype
        scores = self.attn_score(atom_out).squeeze(-1)  # (n_atoms,)

        scores_max = torch.full((n_molecules,), float("-inf"), device=device, dtype=dtype).scatter_reduce(
            0, molecule_id, scores, reduce="amax", include_self=False
        )
        shifted = (scores - scores_max[molecule_id]).exp()
        denom = torch.zeros(n_molecules, device=device, dtype=dtype).index_add(0, molecule_id, shifted).clamp(min=1e-12)
        weights = shifted / denom[molecule_id]

        weighted = atom_out * weights.unsqueeze(-1)
        return torch.zeros(n_molecules, atom_out.shape[-1], device=device, dtype=dtype).index_add(0, molecule_id, weighted)


class PoolingGNN(nn.Module):
    """Identical architecture to the training script's PoolingGNN -- must
    match exactly for pooler_checkpoint.pt's state_dict to load correctly.
    The readout expands each input group into N_POOL_VIEWS separate groups
    (n_groups -> n_groups * N_POOL_VIEWS) instead of collapsing back to one.
    """

    def __init__(self, embed_dim: int, n_layers: int, dropout: float, n_heads: int = 1):
        super().__init__()
        self.convs = nn.ModuleList(
            MultiAggregatorConv(embed_dim, embed_dim, n_heads=n_heads) for _ in range(n_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(embed_dim) for _ in range(n_layers))
        self.dropout = nn.Dropout(dropout)
        self.pool = MultiViewPool(embed_dim)

    def forward(
        self,
        graph: dgl.DGLGraph,
        atom_embeddings_grouped: torch.Tensor,
        molecule_id: torch.Tensor,
        n_molecules: int,
    ) -> torch.Tensor:
        n_atoms, n_groups, embed_dim = atom_embeddings_grouped.shape
        dtype = atom_embeddings_grouped.dtype

        pooled_views_per_group = []
        for g in range(n_groups):
            h = atom_embeddings_grouped[:, g, :].float()
            for conv, norm in zip(self.convs, self.norms):
                h = self.dropout(F.relu(norm(conv(graph, h))))
            pooled_views_per_group.append(self.pool(h, molecule_id, n_molecules))

        pooled = torch.stack(pooled_views_per_group, dim=1)  # (n_molecules, n_groups, N_POOL_VIEWS, embed_dim)
        pooled = pooled.reshape(n_molecules, n_groups * N_POOL_VIEWS, embed_dim)
        return pooled.to(dtype)


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


@dataclass
class SampledDataset:
    graph: dgl.DGLGraph
    atom_embeddings_grouped: torch.Tensor
    molecule_id: torch.Tensor
    n_molecules: int
    y_norm: torch.Tensor
    eval_pos_molecules: int
    conv_type: str


@contextmanager
def _temporary_rng_seed(seed: int):
    """Saves/restores the torch+numpy global RNG state so sampling one
    deterministic dataset never perturbs the sequence used for the next one
    (mirrors the training scripts' identical helper).
    """
    torch_state = torch.random.get_rng_state()
    np_state = np.random.get_state()
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(torch_state)
        np.random.set_state(np_state)


def _sample_prior_dataset(prior_config: dict) -> tuple[torch.Tensor, dgl.DGLGraph, torch.Tensor, str]:
    """Same body as lib/graphpfn/prior/graph_level.py's
    sample_graph_level_dataset, but also returns the resolved conv_type
    (here it's forced by build_prior_config_for_conv_type, but recovering it
    from `config` -- rather than just trusting the caller's intent -- is a
    cheap correctness check against a prior-config-construction bug).
    """
    configs = sample_configs(prior_config, batch_size=1)
    config = configs[0]["prior"]

    graph = sample_multi_graph(**unpack(config["graph"]["sampler"]))
    atom_features, y_per_molecule = sample_graph_level_labels_via_virtual_node(graph, config["scm"])
    atom_features = process_features(
        atom_features,
        p_cat=config["postprocessing"]["p_cat"],
        max_categories=config["postprocessing"]["max_categories"],
        do_permute_features=config["postprocessing"]["permute_features"],
    )
    return atom_features, graph, y_per_molecule, config["scm"]["conv_type"]


def _sample_raw_dataset(features_per_group: int, prior_config: dict) -> dict:
    """CPU-bound, model-free dataset sampling + context-first reorder +
    feature padding -- identical logic to the training script's
    _sample_raw_dataset (see its docstring), returned as a plain dict of CPU
    tensors so it's safe to run in a multiprocessing worker.
    """
    atom_features, graph, y_per_molecule, conv_type = _sample_prior_dataset(prior_config)

    graph.ndata["feat"] = atom_features
    mol_graphs = dgl.unbatch(graph)
    n_molecules = len(mol_graphs)
    perm = np.random.permutation(n_molecules)
    n_train = int(n_molecules * TRAIN_FRACTION)
    mol_order = np.concatenate([perm[:n_train], perm[n_train:]])

    reordered_graph = dgl.batch([mol_graphs[i] for i in mol_order])
    atom_features_reordered = reordered_graph.ndata["feat"]
    counts_reordered = reordered_graph.batch_num_nodes()
    molecule_id = torch.repeat_interleave(torch.arange(n_molecules), counts_reordered)
    eval_pos_atoms = int(counts_reordered[:n_train].sum().item())

    y_reordered = y_per_molecule[torch.from_numpy(mol_order)]
    y_mean = y_reordered[:n_train].mean()
    y_std = y_reordered[:n_train].std().clamp(min=1e-6)
    y_norm = (y_reordered - y_mean) / y_std

    n_atoms, n_features = atom_features_reordered.shape
    feature_to_add = n_features % features_per_group
    if feature_to_add > 0:
        pad = torch.zeros(n_atoms, features_per_group - feature_to_add, dtype=atom_features_reordered.dtype)
        atom_features_reordered = torch.cat([atom_features_reordered, pad], dim=-1)

    src, dst = reordered_graph.edges()
    return {
        "atom_features": atom_features_reordered,
        "edges_src": src,
        "edges_dst": dst,
        "n_atoms": n_atoms,
        "molecule_id": molecule_id,
        "n_molecules": n_molecules,
        "y_norm": y_norm,
        "eval_pos_atoms": eval_pos_atoms,
        "eval_pos_molecules": n_train,
        "conv_type": conv_type,
    }


def _sample_diagnostic_raw_dataset(
    conv_type: str, min_molecules: int, max_molecules: int, features_per_group: int, seed: int
) -> dict:
    """Module-level (picklable) worker function for multiprocessing.Pool --
    builds this conv_type's forced prior_config itself (cheap) rather than
    receiving it as an argument, so nothing DGL-graph-shaped needs to cross
    the process boundary except the returned raw dict.
    """
    prior_config = build_prior_config_for_conv_type(min_molecules, max_molecules, conv_type)
    with _temporary_rng_seed(seed):
        return _sample_raw_dataset(features_per_group, prior_config)


def encode_raw_dataset_on_gpu(model: nn.Module, device: torch.device, raw: dict) -> SampledDataset:
    """Identical to the training script's encode_raw_dataset_on_gpu -- the
    GPU half of dataset prep (frozen encoder_x forward pass)."""
    n_atoms = raw["n_atoms"]
    n_molecules = raw["n_molecules"]
    graph = dgl.graph((raw["edges_src"], raw["edges_dst"]), num_nodes=n_atoms).to(device)
    atom_features = raw["atom_features"].to(device)
    molecule_id = raw["molecule_id"].to(device)
    y_norm = raw["y_norm"].to(device)
    eval_pos_atoms = raw["eval_pos_atoms"]

    features_per_group = model.features_per_group
    n_groups = atom_features.shape[-1] // features_per_group

    x = atom_features.unsqueeze(0)
    x_dict = {"data": x, "mask": torch.isnan(x).to(torch.int32)}
    x_dict = {k: v.reshape(1, n_atoms, n_groups, features_per_group) for k, v in x_dict.items()}
    x_dict["eval_pos"] = eval_pos_atoms
    with torch.no_grad():
        preprocessed = model.x_preprocess(x_dict)
        preprocessed = model.process_4_x(preprocessed)
        x_encoder_result = model.encoder_x(preprocessed)
    atom_embeddings_grouped = x_encoder_result["data"].squeeze(0)

    return SampledDataset(
        graph=graph,
        atom_embeddings_grouped=atom_embeddings_grouped,
        molecule_id=molecule_id,
        n_molecules=n_molecules,
        y_norm=y_norm,
        eval_pos_molecules=raw["eval_pos_molecules"],
        conv_type=raw["conv_type"],
    )


def forward_pass(model: nn.Module, dataset: SampledDataset, pooler_module: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Identical to the training script's forward_pass."""
    eval_pos = dataset.eval_pos_molecules
    pooled_grouped = pooler_module(dataset.graph, dataset.atom_embeddings_grouped, dataset.molecule_id, dataset.n_molecules)
    pooled_x = pooled_grouped.unsqueeze(0).to(next(model.encoder_x.parameters()).dtype)

    embedded_x = model.add_embeddings(pooled_x)

    y_local = dataset.y_norm.unsqueeze(0).unsqueeze(-1).clone()
    y_dict = {"data": y_local}
    y_dict["data"][:, eval_pos:] = torch.nan
    y_type = torch.ones_like(y_dict["data"])
    embedded_y = model.mixed_y_embedding(y_dict, y_type=y_type, eval_pos=eval_pos)

    embedded_all = torch.cat((embedded_x, embedded_y.unsqueeze(2).to(embedded_x.dtype)), dim=2)
    encoder_out = model.transformer_encoder(embedded_all, feature_atten_mask=None, eval_pos=eval_pos)[0]
    encoder_out = model.encoder_out_norm(encoder_out)

    test_encoder_out = encoder_out[:, eval_pos:, -1]
    test_y_type = y_type[:, eval_pos:]
    _, reg_output = model.y_decoder(test_encoder_out, test_y_type)
    pred = reg_output.float().squeeze(0).squeeze(-1)
    target = dataset.y_norm[eval_pos:]
    return pred, target


def summarize(records: list[dict]) -> None:
    """Prints, per conv_type: every (n_molecules, r2) sample sorted by size,
    plus a coarse small/medium/large-tercile breakdown and a Pearson
    correlation between n_molecules and R2 -- a negative, non-trivial
    correlation supports "this conv_type gets harder as molecules grow";
    a near-zero correlation with scattered good/bad points instead supports
    "a few unlucky hard instances," not a systematic size effect.
    """
    for conv_type in CONV_TYPES:
        rows = sorted((r for r in records if r["conv_type"] == conv_type), key=lambda r: r["n_molecules"])
        if not rows:
            continue
        print(f"\n=== {conv_type} ({len(rows)} samples) ===")
        for r in rows:
            print(f"  n_molecules={r['n_molecules']:5d}  R2={r['r2']:+.4f}")

        sizes = np.array([r["n_molecules"] for r in rows], dtype=np.float64)
        r2s = np.array([r["r2"] for r in rows], dtype=np.float64)

        n_bucket = max(1, len(rows) // 3)
        small, medium, large = rows[:n_bucket], rows[n_bucket:-n_bucket] or rows[n_bucket:], rows[-n_bucket:]
        for name, bucket in [("small", small), ("medium", medium), ("large", large)]:
            if bucket:
                mean_r2 = np.mean([r["r2"] for r in bucket])
                mean_n = np.mean([r["n_molecules"] for r in bucket])
                print(f"  {name:6s}: mean n_molecules={mean_n:7.0f}  mean R2={mean_r2:+.4f}")

        if len(rows) >= 3 and sizes.std() > 0 and r2s.std() > 0:
            corr = float(np.corrcoef(sizes, r2s)[0, 1])
            print(f"  corr(n_molecules, R2) = {corr:+.3f}  (more negative -> more support for a real size effect)")


def main() -> None:
    args = parse_args()
    output_json = args.output_json or (args.checkpoint_path.parent / "conv_type_size_diagnostic.json")

    # Load on CPU first and run ALL multiprocessing sampling before the model
    # touches CUDA (defense in depth). The actual fix for the hang this hit
    # in practice is using the 'spawn' start method below, not this ordering
    # -- `import torch` alone (regardless of CUDA) starts background threads
    # (intra-op parallelism, BLAS/OpenMP) that can hold internal locks at the
    # moment fork() is called, deadlocking the child forever. 'spawn' starts
    # each worker as a genuinely fresh interpreter, sidestepping inherited
    # locks entirely, at the cost of a few seconds' re-import overhead per
    # worker (once, not per-sample).
    print(f"Loading real LimiX-16M checkpoint (CPU) from {CHECKPOINT_PATH}...")
    model = load_model(str(CHECKPOINT_PATH), mask_prediction=False)
    for p in model.parameters():
        p.requires_grad = False
    embed_dim = model.embed_dim
    features_per_group = model.features_per_group
    print(f"features_per_group={features_per_group}, embed_dim={embed_dim}")

    print(
        f"Sampling {args.n_samples_per_conv_type} held-out datasets per conv_type "
        f"({args.min_molecules}-{args.max_molecules} molecules, log-uniform), {args.n_workers} workers "
        f"(before touching the GPU -- see comment above)..."
    )
    jobs = [
        (conv_type, args.min_molecules, args.max_molecules, features_per_group, args.seed_base + i * 100_000 + j)
        for i, conv_type in enumerate(CONV_TYPES)
        for j in range(args.n_samples_per_conv_type)
    ]
    spawn_ctx = multiprocessing.get_context("spawn")
    with spawn_ctx.Pool(args.n_workers) as pool:
        raw_datasets = pool.starmap(_sample_diagnostic_raw_dataset, jobs)
    print(f"Sampled {len(raw_datasets)} raw datasets.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = model.to(device)

    print(f"Loading trained pooler checkpoint from {args.checkpoint_path}...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    pooler = PoolingGNN(embed_dim=embed_dim, n_layers=N_GNN_LAYERS, dropout=DROPOUT, n_heads=N_ATTN_HEADS).to(device)
    ema_multi_avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(decay=EMA_DECAY)
    pooler_ema = torch.optim.swa_utils.AveragedModel(pooler, device, multi_avg_fn=ema_multi_avg_fn)
    pooler_ema.load_state_dict(checkpoint["pooler_ema"])
    pooler_ema.eval()
    print(f"Loaded checkpoint from step {checkpoint['step']} (best held-out R2 = {checkpoint['best_held_out_r2']:.4f})")

    records = []
    with torch.no_grad():
        for raw in raw_datasets:
            dataset = encode_raw_dataset_on_gpu(model, device, raw)
            pred, target = forward_pass(model, dataset, pooler_ema)
            records.append(
                {"conv_type": dataset.conv_type, "n_molecules": dataset.n_molecules, "r2": r2_score(pred, target)}
            )

    summarize(records)

    with open(output_json, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nRaw records saved to {output_json}")


if __name__ == "__main__":
    main()
