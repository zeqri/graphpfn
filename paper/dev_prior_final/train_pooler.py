"""Trains the graph pooler (PoolingGNN + per-block RealGraphAdapter) on top of the frozen LimiX-16M
backbone, using synthetic molecule datasets from the GraphPFN prior.

PRIOR (build_prior_config): molecule-skeleton graphs with topology calibrated to real ZINC (molecule
size, bond density, ring-size mix), a causal MLP-SCM producing atom features + one regression label
per molecule, n_causes drawn from the GraphPFN paper's distribution, randomized conv_type /
graph_conv_ratio, and degree / pagerank structural features each switched on with probability 0.5.

TRAINING: DDP (torchrun) or single process, bf16 autocast, gradient accumulation, AdamW + cosine
schedule with warmup, EMA of the pooler weights. Synthetic datasets are sampled by background
DataLoader workers.

EVALUATION every EVAL_EVERY steps (EMA weights):
  * synthetic held-out: mean R2 over N_EVAL_DATASETS fixed-seed prior draws (diagnostic only);
  * real ZINC-12k: --zinc-context train molecules as context, the val split in groups of
    --zinc-query as query. The lowest ZINC MAE selects pooler_checkpoint_best.pt.
ZINC is read from --zinc-root (default <repo>/datasets/zinc; torch_geometric downloads it there if
missing -- do that once from a node with internet before a multi-process run).

OUTPUT: output/train_pooler/<run>/ next to this file -- pooler_checkpoint.pt (resume),
pooler_checkpoint_best.pt (pass to evaluation/ via --pooler-checkpoint), training_log.jsonl. An
existing pooler_checkpoint.pt in the run directory is resumed automatically.

Usage:
    torchrun --nproc_per_node=4 dev_prior_final/train_pooler.py \\
        --min-molecules 1000 --max-molecules 2000 --n-sampler-workers 11 --prefetch-factor 4 \\
        --n-steps 10000 --lr 0.001 --zinc-context 2000 --zinc-query 500
(Plain `python` runs single-process on one GPU/CPU.)
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
for _p in (str(PAPER_DIR), str(SCRIPT_DIR)):  # SCRIPT_DIR: prior_config.py lives next to this file
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dgl  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.parallel import DistributedDataParallel  # noqa: E402
from torch_geometric.datasets import ZINC  # noqa: E402
from tqdm import tqdm  # noqa: E402

import lib  # noqa: E402
import lib.deep  # noqa: E402
import lib.graphpfn.prior.graphs.molecule_skeleton as molecule_skeleton  # noqa: E402
from lib.graphpfn.model import (  # noqa: E402
    GraphPFNGraphAttentionModule,
    GraphPFNMLPModule,
    GraphPFNResidualModule,
)
from lib.graphpfn.prior.attributes import sample_graph_level_labels_via_virtual_node  # noqa: E402
from lib.graphpfn.prior.config import sample_configs  # noqa: E402
from lib.graphpfn.prior.graphs.molecule_skeleton import _sample_bond_order  # noqa: E402
from lib.graphpfn.prior.graphs.multi_graph import sample_multi_graph  # noqa: E402
from lib.graphpfn.prior.graphs.tree_with_rings import _bfs_within_distance  # noqa: E402
from lib.graphpfn.prior.postprocessing import process_features  # noqa: E402
from lib.graphpfn.prior.prior_typings import unpack  # noqa: E402
from vendor.limix.utils.loading import load_model  # noqa: E402
from prior_config import BASE_PRIOR_CONFIG  # noqa: E402

SEED = 0
CHECKPOINT_PATH = PAPER_DIR / "checkpoints" / "LimiX-16M.ckpt"
TRAIN_FRACTION = 0.8  # synthetic training draws only -- ZINC eval uses --zinc-context/--zinc-query directly.

CONV_TYPES = ["gcn", "sage-mean", "sage-min", "sage-max", "gt"]
GRAPH_CONV_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Molecule-skeleton topology calibrated to real ZINC: mean molecule size, bond density, ring-size mix (%).
CALIBRATED_BASE_N_NODES = 55
CALIBRATED_AVG_DEGREE = 2.15
REAL_RING_SIZE_PCT = {3: 2.28, 4: 0.60, 5: 30.71, 6: 65.00, 7: 1.18}

# use_degree / use_pagerank each switched on with probability 0.5 (GraphPFN paper, Section 4.2).
STRUCTURAL_FLAG_VALUES = [True, False]

# n_causes distribution of the GraphPFN paper (exp/graphpfn/pretrain/main/pretrain.toml).
PAPER_N_CAUSES = {
    "_distribution_": "meta_trunc_norm_log_scaled",
    "min_mean": 1, "max_mean": 12, "min_std": 0.01, "max_std": 1.0, "lower_bound": 1, "round": True,
}

N_GNN_LAYERS = 1
N_ATTN_HEADS = 4
N_POOL_VIEWS = 4
DROPOUT = 0.0
GRAPH_ADAPTER_N_HEADS = 4

N_GRADIENT_ACCUMULATION_STEPS = 20
OPTIMIZER_TYPE = "AdamW"
WEIGHT_DECAY = 0.1
GRADIENT_CLIPPING_NORM = 1.0
EMA_DECAY = 0.98
LR_SCHEDULER = "cosine"
WARMUP_FRACTION = 0.1

EVAL_EVERY = 20
N_EVAL_DATASETS = 8  # synthetic held-out eval: matches pretrain.toml's evaluation_data.n_synthetic_per_gpu.
EVAL_SEED_BASE = 10_000  # synthetic held-out eval: disjoint from every training draw's RNG range.
FETCH_STALL_WARN_S = 1.0

DEFAULT_ZINC_ROOT = PAPER_DIR.parent / "datasets" / "zinc"  # <repo>/datasets/zinc
ZINC_EVAL_SEED = 10_000  # fixes the one-time partition of the ZINC val pool into query groups.

OUTPUT_ROOT = SCRIPT_DIR / "output" / "train_pooler"


def _autocast_ctx(device: torch.device):
    """torch.autocast(bfloat16) on the given device."""
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zinc-root", type=Path, default=DEFAULT_ZINC_ROOT,
                        help="torch_geometric ZINC root for the real-ZINC held-out eval; downloaded there by "
                        "torch_geometric if missing (needs internet) (default: %(default)s).")
    parser.add_argument("--min-molecules", type=int, default=300, help="Synthetic training draws: log_uniform_int lower bound (default: 300).")
    parser.add_argument("--max-molecules", type=int, default=3000, help="Synthetic training draws: log_uniform_int upper bound (default: 3000).")
    parser.add_argument("--n-sampler-workers", type=int, default=6, help="Background CPU workers/rank prefetching synthetic training datasets (default: 6).")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="DataLoader prefetch_factor per worker (default: 4).")
    parser.add_argument("--n-steps", type=int, default=1000, help="Total optimizer steps (default: 1000). Sizes run_output_dir.")
    parser.add_argument("--lr", type=float, default=0.001, help="Peak LR, cosine+10%% warmup (default: 0.001). Sizes run_output_dir.")
    parser.add_argument(
        "--zinc-context", type=int, default=2000,
        help="Real-ZINC held-out eval: context molecules drawn fresh from the ZINC TRAIN pool per "
        "group, per eval round (default: %(default)s).",
    )
    parser.add_argument(
        "--zinc-query", type=int, default=500,
        help="Real-ZINC held-out eval: fixed group size the ZINC VAL pool is partitioned into "
        "(default: 500 -- with the default 1000-molecule val split, 2 groups, scored and averaged "
        "every eval round).",
    )
    return parser.parse_args()


def _calibrated_ring_closure(
    n_nodes: int, avg_degree: float, valence_cap: np.ndarray, *, min_ring_size: int, max_ring_size: int,
) -> dict[tuple[int, int], int]:
    """Replacement for molecule_skeleton._sample_valence_capped_tree_with_rings (same signature).
    Grows a valence-capped tree, then closes rings: reachable candidates are bucketed by BFS
    distance (ring size = distance + 1), a bucket is picked with probability REAL_RING_SIZE_PCT[size],
    then a candidate uniformly within it.
    """
    remaining_valence = valence_cap.copy()
    adjacency: list[set[int]] = [set() for _ in range(n_nodes)]
    order_by_pair: dict[tuple[int, int], int] = {}

    def add_edge(u: int, v: int) -> None:
        order = _sample_bond_order(remaining_valence[u], remaining_valence[v])
        order_by_pair[(min(u, v), max(u, v))] = order
        adjacency[u].add(v)
        adjacency[v].add(u)
        remaining_valence[u] -= order
        remaining_valence[v] -= order

    for i in range(1, n_nodes):
        candidates = [j for j in range(i) if remaining_valence[j] >= 1]
        if not candidates:
            candidates = list(range(i))
        j = int(np.random.choice(candidates))
        add_edge(i, j)

    target_extra_edges = max(0, round(avg_degree * n_nodes / 2) - (n_nodes - 1))
    max_attempts = target_extra_edges * 20 + 50
    added = 0
    attempts = 0
    while added < target_extra_edges and attempts < max_attempts:
        attempts += 1
        u = int(np.random.randint(n_nodes))
        if remaining_valence[u] < 1:
            continue
        nearby = _bfs_within_distance(adjacency, u, max_ring_size - 1)
        candidates = [
            v for v, d in nearby.items()
            if d >= min_ring_size - 1 and remaining_valence[v] >= 1 and v not in adjacency[u]
        ]
        if not candidates:
            continue
        by_distance: dict[int, list[int]] = {}
        for v in candidates:
            by_distance.setdefault(nearby[v], []).append(v)
        bucket_distances = list(by_distance.keys())
        bucket_weights = np.array([REAL_RING_SIZE_PCT.get(d + 1, 1e-6) for d in bucket_distances])
        bucket_weights = bucket_weights / bucket_weights.sum()
        chosen_distance = bucket_distances[np.random.choice(len(bucket_distances), p=bucket_weights)]
        v = int(np.random.choice(by_distance[chosen_distance]))
        add_edge(u, v)
        added += 1

    return order_by_pair


def _install_calibrated_ring_closure() -> None:
    """Installs _calibrated_ring_closure into molecule_skeleton. Must run before any sampling, on
    every rank and in every prefetch worker process.
    """
    molecule_skeleton._sample_valence_capped_tree_with_rings = _calibrated_ring_closure


def build_prior_config(min_molecules: int, max_molecules: int) -> dict:
    """BASE_PRIOR_CONFIG with: molecules per dataset ~ log_uniform_int(min, max); randomized
    conv_type / graph_conv_ratio; ZINC-calibrated topology; causal SCM; paper n_causes distribution;
    randomized structural features.
    """
    prior_config = copy.deepcopy(BASE_PRIOR_CONFIG)
    prior_values = prior_config["prior"]["values"][0]

    n_graphs_dist = {"_distribution_": "log_uniform_int", "min": min_molecules, "max": max_molecules}
    prior_values["graph"]["sampler"]["values"][0]["n_graphs"] = n_graphs_dist
    prior_values["graph"]["n_nodes"] = n_graphs_dist

    prior_values["scm"]["conv_type"] = {"_distribution_": "choice", "values": CONV_TYPES}
    prior_values["scm"]["graph_conv_ratio"] = {"_distribution_": "choice", "values": GRAPH_CONV_RATIOS}

    prior_values["graph"]["sampler"]["values"][0]["base_n_nodes"] = {
        "_distribution_": "choice", "values": [CALIBRATED_BASE_N_NODES]
    }
    prior_values["graph"]["sampler"]["values"][0]["sub_graph"]["avg_degree"] = {
        "_distribution_": "choice", "values": [CALIBRATED_AVG_DEGREE]
    }

    prior_values["scm"]["base"]["causal"]["enabled"] = {"_distribution_": "choice", "values": [True]}

    prior_values["scm"]["base"]["n_causes"] = PAPER_N_CAUSES

    prior_values["scm"]["structural"]["use_degree"] = {
        "_distribution_": "choice", "values": STRUCTURAL_FLAG_VALUES
    }
    prior_values["scm"]["structural"]["use_pagerank"] = {
        "_distribution_": "choice", "values": STRUCTURAL_FLAG_VALUES
    }

    return prior_config


def build_run_paths(min_molecules: int, max_molecules: int, zinc_context: int, zinc_query: int, n_steps: int, lr: float) -> tuple[Path, Path, Path, Path]:
    """The run directory name encodes every setting that changes training, so a different config
    starts a fresh run instead of resuming an incompatible one. Returns
    (run_dir, resume_checkpoint_path, best_checkpoint_path, log_path).
    """
    run_output_dir = OUTPUT_ROOT / (
        f"molecules_{min_molecules}_{max_molecules}_zinc_{zinc_context}_{zinc_query}_"
        f"n_steps_{n_steps}_lr_{lr:g}"
    )
    return (
        run_output_dir,
        run_output_dir / "pooler_checkpoint.pt",
        run_output_dir / "pooler_checkpoint_best.pt",
        run_output_dir / "training_log.jsonl",
    )


class MultiAggregatorConv(nn.Module):
    """Per-layer message passing combining mean/min/max neighbor reductions with a lightweight
    multi-head dot-product attention aggregation.
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
    """Atom->molecule readout emitting mean/min/max/attention pooling as separate groups."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.attn_score = nn.Linear(embed_dim, 1)

    def forward(self, atom_out: torch.Tensor, molecule_id: torch.Tensor, n_molecules: int) -> torch.Tensor:
        mean_pool = _scatter_pool(atom_out, molecule_id, n_molecules, reduce="mean", init_value=0.0)
        min_pool = _scatter_pool(atom_out, molecule_id, n_molecules, reduce="amin", init_value=float("inf"))
        max_pool = _scatter_pool(atom_out, molecule_id, n_molecules, reduce="amax", init_value=float("-inf"))
        attn_pool = self._attention_pool(atom_out, molecule_id, n_molecules)
        return torch.stack([mean_pool, min_pool, max_pool, attn_pool], dim=1)

    def _attention_pool(self, atom_out: torch.Tensor, molecule_id: torch.Tensor, n_molecules: int) -> torch.Tensor:
        device, dtype = atom_out.device, atom_out.dtype
        # atom_out comes from DGL ops (not autocast) while attn_score is autocast -- match dtypes,
        # which scatter_reduce requires.
        scores = self.attn_score(atom_out).squeeze(-1).to(dtype)

        scores_max = torch.full((n_molecules,), float("-inf"), device=device, dtype=dtype).scatter_reduce(
            0, molecule_id, scores, reduce="amax", include_self=False
        )
        shifted = (scores - scores_max[molecule_id]).exp()
        denom = torch.zeros(n_molecules, device=device, dtype=dtype).index_add(0, molecule_id, shifted).clamp(min=1e-12)
        weights = shifted / denom[molecule_id]

        weighted = atom_out * weights.unsqueeze(-1)
        return torch.zeros(n_molecules, atom_out.shape[-1], device=device, dtype=dtype).index_add(0, molecule_id, weighted)


class RealGraphAdapter(nn.Module):
    """One transformer block's graph-attention + MLP adapter (modules from lib.graphpfn.model)."""

    def __init__(self, embed_dim: int, n_heads: int = GRAPH_ADAPTER_N_HEADS, dropout: float = 0.0, zero_init: bool = True):
        super().__init__()
        self.conv = GraphPFNResidualModule(
            base=GraphPFNGraphAttentionModule(d=embed_dim, n_heads=n_heads, dropout=dropout, zero_init=zero_init),
            d_hidden=embed_dim,
        )
        self.mlp = GraphPFNResidualModule(
            base=GraphPFNMLPModule(d=embed_dim, zero_init=zero_init),
            d_hidden=embed_dim,
        )

    def forward(self, augmented_graph: dgl.DGLGraph, combined_hidden: torch.Tensor) -> torch.Tensor:
        n_total, embed_dim = combined_hidden.shape
        x = combined_hidden.reshape(1, n_total, 1, embed_dim)
        x = self.conv(augmented_graph, x)
        x = self.mlp(augmented_graph, x)
        return x.reshape(n_total, embed_dim)


class PoolingGNN(nn.Module):
    """Message passing over bonds + atom->molecule pooling + per-block RealGraphAdapter refinement
    of each molecule's virtual node inside the frozen backbone.
    """

    def __init__(self, embed_dim: int, n_layers: int, dropout: float, n_transformer_blocks: int, n_heads: int = 1):
        super().__init__()
        self.convs = nn.ModuleList(
            MultiAggregatorConv(embed_dim, embed_dim, n_heads=n_heads) for _ in range(n_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(embed_dim) for _ in range(n_layers))
        self.dropout = nn.Dropout(dropout)
        self.pool = MultiViewPool(embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        self.graph_adapters = nn.ModuleList(
            RealGraphAdapter(embed_dim, n_heads=GRAPH_ADAPTER_N_HEADS, dropout=dropout)
            for _ in range(n_transformer_blocks)
        )

    def forward(
        self, model: nn.Module, graph: dgl.DGLGraph, augmented_graph: dgl.DGLGraph,
        atom_embeddings_grouped: torch.Tensor, molecule_id: torch.Tensor, n_molecules: int,
        embedded_y: torch.Tensor, eval_pos: int,
    ) -> torch.Tensor:
        n_atoms, n_groups, embed_dim = atom_embeddings_grouped.shape
        assert len(self.graph_adapters) == len(model.transformer_encoder.layers), (
            f"{len(self.graph_adapters)=} != {len(model.transformer_encoder.layers)=}"
        )

        pooled_views_per_group = []
        atom_hidden_per_group = []
        for g in range(n_groups):
            h = atom_embeddings_grouped[:, g, :].float()
            for conv, norm in zip(self.convs, self.norms):
                h = self.dropout(F.relu(norm(conv(graph, h))))
            atom_hidden_per_group.append(h)
            pooled_views_per_group.append(self.pool(h, molecule_id, n_molecules))

        pooled = torch.stack(pooled_views_per_group, dim=1)
        pooled = pooled.reshape(n_molecules, n_groups * N_POOL_VIEWS, embed_dim)
        pooled = self.output_proj(pooled)

        atom_hidden = torch.stack(atom_hidden_per_group, dim=1).mean(dim=1)
        virtual_node = pooled.float().mean(dim=1)

        combined_hidden = torch.cat([atom_hidden, virtual_node], dim=0)

        add_embeddings_dtype = next(model.encoder_x.parameters()).dtype
        vnode_idx = pooled.shape[1]
        pooled_x = torch.cat([pooled, virtual_node.unsqueeze(1).to(pooled.dtype)], dim=1)
        pooled_x = pooled_x.unsqueeze(0).clone().to(add_embeddings_dtype)

        embedded_x = model.add_embeddings(pooled_x)
        x = torch.cat((embedded_x, embedded_y.unsqueeze(2).to(embedded_x.dtype)), dim=2)

        for i, block in enumerate(model.transformer_encoder.layers):
            x, _, _ = block(x, None, eval_pos, layer_idx=i)

            before = x[:, :, :vnode_idx, :]
            current_vnode = x[:, :, vnode_idx, :].squeeze(0).float()
            after = x[:, :, vnode_idx + 1 :, :]

            combined_hidden = torch.cat([combined_hidden[:n_atoms], current_vnode], dim=0)
            combined_hidden = self.graph_adapters[i](augmented_graph, combined_hidden)

            current_vnode = combined_hidden[n_atoms:]
            new_vnode_slot = current_vnode.to(x.dtype).unsqueeze(0).unsqueeze(2)
            x = torch.cat([before, new_vnode_slot, after], dim=2)

        return x


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """pred/target: (n_query,) standardized continuous floats. Reports R2 AND MAE."""
    return {"r2": r2_score(pred, target), "mae": F.l1_loss(pred, target).item()}


def save_checkpoint(
    path: Path, step: int, pooler_without_ddp: nn.Module, pooler_ema: nn.Module,
    optimizer: torch.optim.Optimizer, lr_scheduler, best_held_out_mae: float, best_step: int,
) -> None:
    """Written to a .tmp path and renamed into place so a crash mid-write never leaves a corrupt
    checkpoint. Called for BOTH the resume checkpoint (every eval round) and the best checkpoint
    (only when held-out MAE improves).
    """
    tmp_path = path.with_suffix(".tmp")
    torch.save(
        {
            "step": step,
            "pooler": pooler_without_ddp.state_dict(),
            "pooler_ema": pooler_ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "best_held_out_mae": best_held_out_mae,
            "best_step": best_step,
        },
        tmp_path,
    )
    tmp_path.replace(path)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device)


def append_json_log(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------------------------------
# Real-ZINC held-out eval helpers
# --------------------------------------------------------------------------------------------------

def _graph_has_isolated_atom(edge_index: torch.Tensor, n_atoms: int) -> bool:
    """True if any atom has zero bonds (its mean/min/max aggregation would reduce to NaN/inf and
    poison every other molecule via the backbone's cross-molecule attention)."""
    in_degree = torch.zeros(n_atoms, dtype=torch.long)
    in_degree.scatter_add_(0, edge_index[1], torch.ones(edge_index.shape[1], dtype=torch.long))
    return bool((in_degree == 0).any())


def _build_zinc_pool(ds: ZINC) -> np.ndarray:
    """Indices of every molecule in a ZINC split that has no isolated atom."""
    keep = []
    for idx in range(len(ds)):
        data = ds[idx]
        if not _graph_has_isolated_atom(data.edge_index, data.x.shape[0]):
            keep.append(idx)
    return np.array(keep)


def _compute_global_target_stats(train_ds: ZINC, train_pool_indices: np.ndarray) -> tuple[float, float]:
    """Fixed mean/std of the target over the filtered ZINC TRAIN split, used for every eval round."""
    y_values = torch.cat([train_ds[int(i)].y.view(1) for i in train_pool_indices]).float()
    y_mean = y_values.mean().item()
    y_std = max(y_values.std().item(), 1e-6)
    return y_mean, y_std


def _zinc_examples_to_raw(
    examples: list, n_context: int, features_per_group: int, y_mean: float, y_std: float,
) -> dict:
    """Context-first list of ZINC Data objects -> raw dict (same contract as _sample_raw_dataset).
    Atom features = ZINC's single atom-type column as float, no one-hot; y standardized with the
    fixed y_mean/y_std."""
    atom_features_list: list[torch.Tensor] = []
    edges_src_list: list[torch.Tensor] = []
    edges_dst_list: list[torch.Tensor] = []
    molecule_id_list: list[torch.Tensor] = []
    y_list: list[torch.Tensor] = []

    atom_offset = 0
    eval_pos_atoms = None
    for mol_idx, data in enumerate(examples):
        if mol_idx == n_context:
            eval_pos_atoms = atom_offset
        n_atoms_mol = data.x.shape[0]
        atom_features_list.append(data.x.float())
        edges_src_list.append(data.edge_index[0] + atom_offset)
        edges_dst_list.append(data.edge_index[1] + atom_offset)
        molecule_id_list.append(torch.full((n_atoms_mol,), mol_idx, dtype=torch.long))
        y_list.append(data.y.view(1).float())
        atom_offset += n_atoms_mol
    if eval_pos_atoms is None:
        eval_pos_atoms = atom_offset  # degenerate: n_context == len(examples)

    atom_features = torch.cat(atom_features_list, dim=0)
    edges_src = torch.cat(edges_src_list, dim=0)
    edges_dst = torch.cat(edges_dst_list, dim=0)
    molecule_id = torch.cat(molecule_id_list, dim=0)
    y_per_molecule = torch.cat(y_list, dim=0)

    y_mean_t = torch.tensor(y_mean, dtype=y_per_molecule.dtype)
    y_std_t = torch.tensor(y_std, dtype=y_per_molecule.dtype)
    y_norm = (y_per_molecule - y_mean_t) / y_std_t

    n_atoms, n_features = atom_features.shape
    feature_to_add = n_features % features_per_group
    if feature_to_add > 0:
        pad = torch.zeros(n_atoms, features_per_group - feature_to_add, dtype=atom_features.dtype)
        atom_features = torch.cat([atom_features, pad], dim=-1)

    return {
        "atom_features": atom_features,
        "edges_src": edges_src,
        "edges_dst": edges_dst,
        "n_atoms": n_atoms,
        "molecule_id": molecule_id,
        "n_molecules": len(examples),
        "y_norm": y_norm,
        "eval_pos_atoms": eval_pos_atoms,
        "eval_pos_molecules": n_context,
    }


def _build_eval_query_groups(val_pool_indices: np.ndarray, n_query: int, seed: int) -> list[np.ndarray]:
    """Partitions the filtered ZINC val pool ONCE into fixed groups of n_query molecules (one fixed
    shuffle via `seed`), so every val molecule is scored every eval round."""
    rng = np.random.RandomState(seed)
    shuffled = val_pool_indices.copy()
    rng.shuffle(shuffled)
    return [shuffled[i : i + n_query] for i in range(0, len(shuffled), n_query)]


@dataclass
class SampledDataset:
    graph: dgl.DGLGraph
    augmented_graph: dgl.DGLGraph
    atom_embeddings_grouped: torch.Tensor
    molecule_id: torch.Tensor
    n_molecules: int
    y_norm: torch.Tensor
    eval_pos_molecules: int
    conv_type: str  # "real_zinc" placeholder for ZINC-derived eval datasets.
    graph_conv_ratio: float  # 0.0 placeholder for ZINC-derived eval datasets.


def _sample_prior_dataset_with_conv_type_and_ratio(
    prior_config: dict,
) -> tuple[torch.Tensor, dgl.DGLGraph, torch.Tensor, str, float]:
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

    return atom_features, graph, y_per_molecule, config["scm"]["conv_type"], config["scm"]["graph_conv_ratio"]


def _sample_raw_dataset(features_per_group: int, prior_config: dict) -> dict:
    atom_features, graph, y_per_molecule, conv_type, graph_conv_ratio = (
        _sample_prior_dataset_with_conv_type_and_ratio(prior_config)
    )

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
        "graph_conv_ratio": graph_conv_ratio,
    }


class _RawSyntheticDatasetIterable(torch.utils.data.IterableDataset):
    def __init__(self, seed: int, features_per_group: int, prior_config: dict):
        self.seed = seed
        self.features_per_group = features_per_group
        self.prior_config = prior_config

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        torch.manual_seed(self.seed + worker_id)
        np.random.seed(self.seed + worker_id)
        # Each worker is a fresh process (persistent_workers=True) that doesn't inherit main()'s
        # own runtime monkeypatch state -- install it here too (idempotent).
        _install_calibrated_ring_closure()
        while True:
            yield _sample_raw_dataset(self.features_per_group, self.prior_config)


def build_prefetch_loader(seed: int, features_per_group: int, prior_config: dict, n_workers: int, prefetch_factor: int):
    dataset = _RawSyntheticDatasetIterable(seed=seed, features_per_group=features_per_group, prior_config=prior_config)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=n_workers,
        prefetch_factor=prefetch_factor if n_workers > 0 else None,
        persistent_workers=n_workers > 0, collate_fn=lambda batch: batch[0],
    )
    return iter(loader)


def encode_raw_dataset_on_gpu(model: nn.Module, device: torch.device, raw: dict) -> SampledDataset:
    """Builds the bond graph + atom<->virtual-node graph and runs the frozen LimiX feature encoder.
    Used for both synthetic training draws and real-ZINC eval draws.
    """
    n_atoms = raw["n_atoms"]
    n_molecules = raw["n_molecules"]
    graph = dgl.graph((raw["edges_src"], raw["edges_dst"]), num_nodes=n_atoms).to(device)
    atom_features = raw["atom_features"].to(device)
    molecule_id = raw["molecule_id"].to(device)
    y_norm = raw["y_norm"].to(device)
    eval_pos_atoms = raw["eval_pos_atoms"]

    vnode_global_id = n_atoms + torch.arange(n_molecules, device=device)
    mol_of_atom = vnode_global_id[molecule_id]
    atom_arange = torch.arange(n_atoms, device=device)
    atom_src, atom_dst = graph.edges()
    augmented_src = torch.cat([atom_src, atom_arange, mol_of_atom])
    augmented_dst = torch.cat([atom_dst, mol_of_atom, atom_arange])
    augmented_graph = dgl.graph((augmented_src, augmented_dst), num_nodes=n_atoms + n_molecules).to(device)

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
        graph=graph, augmented_graph=augmented_graph, atom_embeddings_grouped=atom_embeddings_grouped,
        molecule_id=molecule_id, n_molecules=n_molecules, y_norm=y_norm,
        eval_pos_molecules=raw["eval_pos_molecules"], conv_type=raw["conv_type"], graph_conv_ratio=raw["graph_conv_ratio"],
    )


def sample_and_prepare_dataset(model: nn.Module, device: torch.device, prior_config: dict) -> SampledDataset:
    """Synchronous synthetic sampling, used for the fixed-seed held-out eval datasets."""
    raw = _sample_raw_dataset(model.features_per_group, prior_config)
    return encode_raw_dataset_on_gpu(model, device, raw)


@contextmanager
def _temporary_rng_seed(seed: int):
    """Seeds torch+numpy for the block and restores the previous RNG state afterwards."""
    torch_state = torch.random.get_rng_state()
    np_state = np.random.get_state()
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(torch_state)
        np.random.set_state(np_state)


def sample_eval_dataset(model: nn.Module, device: torch.device, idx: int, prior_config: dict) -> SampledDataset:
    """Deterministic synthetic held-out dataset number `idx` (seed EVAL_SEED_BASE + idx)."""
    with _temporary_rng_seed(EVAL_SEED_BASE + idx):
        return sample_and_prepare_dataset(model, device, prior_config)


def forward_pass(model: nn.Module, dataset: SampledDataset, pooler_module: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Regression forward (y_type=1) -> (standardized query predictions, targets). Callers wrap
    it in _autocast_ctx.
    """
    eval_pos = dataset.eval_pos_molecules

    y_local = dataset.y_norm.unsqueeze(0).unsqueeze(-1).clone()
    y_dict = {"data": y_local}
    y_dict["data"][:, eval_pos:] = torch.nan
    y_type = torch.ones_like(y_dict["data"])  # 1 == regression branch
    embedded_y = model.mixed_y_embedding(y_dict, y_type=y_type, eval_pos=eval_pos)

    x = pooler_module(
        model, dataset.graph, dataset.augmented_graph, dataset.atom_embeddings_grouped,
        dataset.molecule_id, dataset.n_molecules, embedded_y, eval_pos,
    )
    encoder_out = model.encoder_out_norm(x)

    test_encoder_out = encoder_out[:, eval_pos:, -1]
    test_y_type = y_type[:, eval_pos:]
    _, reg_output = model.y_decoder(test_encoder_out, test_y_type)

    pred = reg_output.float().squeeze(0).squeeze(-1)  # always fp32 regardless of autocast.
    target = dataset.y_norm[eval_pos:]
    return pred, target


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()

    _install_calibrated_ring_closure()  # MUST run before any sampling call, on EVERY rank.

    prior_config = build_prior_config(args.min_molecules, args.max_molecules)
    run_output_dir, pooler_checkpoint_path, pooler_best_checkpoint_path, training_log_path = build_run_paths(
        args.min_molecules, args.max_molecules, args.zinc_context, args.zinc_query, args.n_steps, args.lr
    )

    if lib.is_ddp():
        lib.configure_ddp()
    rank = lib.get_rank()
    world_size = lib.get_world_size()
    device = lib.get_device()
    is_main = lib.is_master_process()

    if is_main:
        run_output_dir.mkdir(parents=True, exist_ok=True)
    lib.barrier()

    torch.manual_seed(SEED + rank)
    np.random.seed(SEED + rank)

    if is_main:
        print(f"DDP: world_size={world_size}" if lib.is_ddp() else "Single-process run (no torchrun)")
        print(f"Loading real LimiX-16M checkpoint from {CHECKPOINT_PATH}...")
    model = load_model(str(CHECKPOINT_PATH), mask_prediction=False)
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    embed_dim = model.embed_dim
    if is_main:
        print(f"features_per_group={model.features_per_group}, embed_dim={embed_dim}, nlayers={model.nlayers}")
        print(
            f"Training on CALIBRATED (base_n_nodes={CALIBRATED_BASE_N_NODES}, "
            f"avg_degree={CALIBRATED_AVG_DEGREE}, ring-size-biased closure), CAUSAL "
            f"(n_causes paper-randomized: {PAPER_N_CAUSES}) synthetic molecule-skeleton data every micro-step "
            f"({args.min_molecules}-{args.max_molecules} molecules; conv_type in {CONV_TYPES}; "
            f"graph_conv_ratio in {GRAPH_CONV_RATIOS}; use_degree/use_pagerank each randomized "
            f"{STRUCTURAL_FLAG_VALUES}, matching the GraphPFN paper's own Section 4.2 scheme) "
            f"under torch.autocast(bfloat16) -- lr={args.lr}, {N_GRADIENT_ACCUMULATION_STEPS} per "
            f"optimizer step, {args.n_steps} steps total. Evaluating every {EVAL_EVERY} steps on "
            f"BOTH {N_EVAL_DATASETS} synthetic fixed-seed held-out datasets (diagnostic only) AND "
            f"real ZINC (context={args.zinc_context}, query groups of {args.zinc_query} -- drives "
            f"checkpoint selection via MAE)."
        )
        print(f"Run output dir: {run_output_dir}")

    # Real-ZINC held-out eval data -- loaded on every rank (cheap, disk-cached), but only rank 0
    # ever calls evaluate_held_out.
    print(f"Loading real ZINC (12k subset) train/val splits from {args.zinc_root}...") if is_main else None
    zinc_train_ds = ZINC(root=str(args.zinc_root), subset=True, split="train")
    zinc_val_ds = ZINC(root=str(args.zinc_root), subset=True, split="val")
    zinc_train_pool = _build_zinc_pool(zinc_train_ds)
    zinc_val_pool = _build_zinc_pool(zinc_val_ds)
    zinc_y_mean, zinc_y_std = _compute_global_target_stats(zinc_train_ds, zinc_train_pool)
    zinc_eval_groups = _build_eval_query_groups(zinc_val_pool, args.zinc_query, ZINC_EVAL_SEED)
    if is_main:
        print(
            f"  usable ZINC molecules: train={len(zinc_train_pool)}, val={len(zinc_val_pool)} "
            f"({len(zinc_eval_groups)} eval group(s)); target mean={zinc_y_mean:.6f}, std={zinc_y_std:.6f}"
        )

    raw_dataset_iter = build_prefetch_loader(
        seed=SEED + rank * 100_000, features_per_group=model.features_per_group,
        prior_config=prior_config, n_workers=args.n_sampler_workers, prefetch_factor=args.prefetch_factor,
    )

    pooler_without_ddp = PoolingGNN(
        embed_dim=embed_dim, n_layers=N_GNN_LAYERS, dropout=DROPOUT,
        n_transformer_blocks=model.nlayers, n_heads=N_ATTN_HEADS,
    ).to(device)
    pooler = pooler_without_ddp
    if lib.is_ddp():
        ddp_device_ids = [lib.get_local_rank()] if device.type == "cuda" else None
        pooler = DistributedDataParallel(pooler_without_ddp, device_ids=ddp_device_ids)

    params = lib.deep.make_parameter_groups(pooler_without_ddp)
    optimizer = lib.deep.make_optimizer(type=OPTIMIZER_TYPE, lr=args.lr, weight_decay=WEIGHT_DECAY, params=params)
    n_warmup_steps = max(1, round(args.n_steps * WARMUP_FRACTION))
    lr_scheduler = lib.deep.get_lr_scheduler(optimizer, n_warmup_steps=n_warmup_steps, n_steps=args.n_steps, scheduler=LR_SCHEDULER)
    ema_multi_avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(decay=EMA_DECAY)
    pooler_ema = torch.optim.swa_utils.AveragedModel(pooler_without_ddp, device, multi_avg_fn=ema_multi_avg_fn)

    start_step = 1
    best_held_out_mae = float("inf")
    best_step = -1
    if pooler_checkpoint_path.exists():
        checkpoint = load_checkpoint(pooler_checkpoint_path, device)
        pooler_without_ddp.load_state_dict(checkpoint["pooler"])
        pooler_ema.load_state_dict(checkpoint["pooler_ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        best_held_out_mae = checkpoint["best_held_out_mae"]
        best_step = checkpoint["best_step"]
        start_step = checkpoint["step"] + 1
        if is_main:
            print(f"Resumed from {pooler_checkpoint_path} at step {checkpoint['step']} (best held-out MAE so far: {best_held_out_mae:.4f} @ step {best_step})")

    def evaluate_held_out_synthetic() -> float:
        """Mean R2 over N_EVAL_DATASETS fixed-seed synthetic datasets (EMA weights; diagnostic only)."""
        pooler_ema.eval()
        per_dataset = []
        with torch.no_grad(), _autocast_ctx(device):
            for idx in range(N_EVAL_DATASETS):
                dataset = sample_eval_dataset(model, device, idx, prior_config)
                pred, target = forward_pass(model, dataset, pooler_ema)
                score = r2_score(pred, target)
                per_dataset.append((dataset.conv_type, dataset.graph_conv_ratio, dataset.n_molecules, score))
        breakdown = " | ".join(
            f"{conv_type}/ratio={graph_conv_ratio:g}(n={n_molecules})=r2:{score:.3f}"
            for conv_type, graph_conv_ratio, n_molecules, score in per_dataset
        )
        tqdm.write(f"  synthetic held-out per-dataset: {breakdown}")
        return float(np.mean([score for *_, score in per_dataset]))

    def evaluate_held_out_zinc() -> tuple[float, float]:
        """Real-ZINC (R2, MAE) averaged over the val query groups (EMA weights); MAE selects the best checkpoint."""
        pooler_ema.eval()
        r2_list, mae_list = [], []
        with torch.no_grad(), _autocast_ctx(device):
            for group_indices in zinc_eval_groups:
                context_indices = np.random.choice(zinc_train_pool, size=args.zinc_context, replace=False)
                examples = [zinc_train_ds[int(i)] for i in context_indices] + [zinc_val_ds[int(i)] for i in group_indices]
                raw = _zinc_examples_to_raw(
                    examples, n_context=len(context_indices), features_per_group=model.features_per_group,
                    y_mean=zinc_y_mean, y_std=zinc_y_std,
                )
                raw["conv_type"] = "real_zinc"  # placeholder, synthetic-only field.
                raw["graph_conv_ratio"] = 0.0  # placeholder.
                eval_dataset = encode_raw_dataset_on_gpu(model, device, raw)
                pred, target = forward_pass(model, eval_dataset, pooler_ema)
                metrics = compute_metrics(pred, target)
                r2_list.append(metrics["r2"])
                mae_list.append(metrics["mae"])
        return float(np.mean(r2_list)), float(np.mean(mae_list))

    if start_step == 1:
        # Baseline eval BEFORE any training (only on a fresh run). Doesn't affect
        # best_held_out_mae/checkpoint selection -- "best" should mean best after some actual
        # training, not the untrained random init.
        if is_main:
            tqdm.write("Evaluating at step 0 (baseline, before any training)...")
            synthetic_held_out_r2_0 = evaluate_held_out_synthetic()
            zinc_held_out_r2_0, zinc_held_out_mae_0 = evaluate_held_out_zinc()
            tqdm.write(
                f"step    0 | held-out (EMA): synthetic R2={synthetic_held_out_r2_0:.4f}  |  "
                f"real ZINC R2={zinc_held_out_r2_0:.4f} MAE={zinc_held_out_mae_0:.4f}"
            )
            append_json_log(
                training_log_path,
                {
                    "step": 0, "loss": None, "grad_norm": None, "lr": lib.deep.get_lr(optimizer),
                    "synthetic_held_out_r2": synthetic_held_out_r2_0,
                    "zinc_held_out_r2": zinc_held_out_r2_0,
                    "zinc_held_out_mae": zinc_held_out_mae_0,
                },
            )
        lib.barrier()

    iterator = range(start_step, args.n_steps + 1)
    if is_main:
        iterator = tqdm(iterator, desc="training", initial=start_step - 1, total=args.n_steps)

    for step in iterator:
        pooler.train()
        optimizer.zero_grad()
        step_losses = []
        for inner_step in range(N_GRADIENT_ACCUMULATION_STEPS):
            is_last_microstep = (inner_step + 1) == N_GRADIENT_ACCUMULATION_STEPS
            t0 = time.time()
            with pooler.no_sync() if (lib.is_ddp() and not is_last_microstep) else nullcontext():
                raw = next(raw_dataset_iter)
                t1 = time.time()
                with _autocast_ctx(device):
                    dataset = encode_raw_dataset_on_gpu(model, device, raw)
                    pred, target = forward_pass(model, dataset, pooler)
                    loss = F.mse_loss(pred, target)
                # backward() OUTSIDE the autocast context -- standard practice, autograd replays
                # whatever dtypes autocast recorded during the forward above.
                (loss / N_GRADIENT_ACCUMULATION_STEPS).backward()
                step_losses.append(loss.detach())
            fetch_time = t1 - t0
            if is_main and fetch_time > FETCH_STALL_WARN_S:
                total_microsteps_done = (step - 1) * N_GRADIENT_ACCUMULATION_STEPS + inner_step + 1
                tqdm.write(
                    f"[stall] micro-step {total_microsteps_done} (step {step}, {inner_step + 1}/"
                    f"{N_GRADIENT_ACCUMULATION_STEPS}) waited fetch={fetch_time:.2f}s -- workers "
                    f"aren't keeping up; consider raising --n-sampler-workers/--prefetch-factor."
                )

        grad_norm = torch.nn.utils.clip_grad_norm_(pooler.parameters(), GRADIENT_CLIPPING_NORM)
        optimizer.step()
        pooler_ema.update_parameters(pooler_without_ddp)
        lr_scheduler.step()
        step_loss_mean = torch.stack(step_losses).mean()

        if is_main:
            if isinstance(iterator, tqdm):
                iterator.set_postfix(loss=step_loss_mean.item(), grad_norm=grad_norm.item(), lr=lib.deep.get_lr(optimizer))
            tqdm.write(f"step {step:5d} | loss {step_loss_mean.item():.4f} | grad_norm {grad_norm.item():.4f} | lr {lib.deep.get_lr(optimizer):.6f}")

        synthetic_held_out_r2 = None
        zinc_held_out_r2 = None
        zinc_held_out_mae = None
        if step % EVAL_EVERY == 0 or step == args.n_steps:
            if is_main:
                synthetic_held_out_r2 = evaluate_held_out_synthetic()
                zinc_held_out_r2, zinc_held_out_mae = evaluate_held_out_zinc()
                tqdm.write(
                    f"step {step:5d} | held-out (EMA): synthetic R2={synthetic_held_out_r2:.4f}  |  "
                    f"real ZINC R2={zinc_held_out_r2:.4f} MAE={zinc_held_out_mae:.4f}"
                )
                save_checkpoint(
                    pooler_checkpoint_path, step, pooler_without_ddp, pooler_ema, optimizer,
                    lr_scheduler, best_held_out_mae, best_step,
                )
                if zinc_held_out_mae < best_held_out_mae:
                    best_held_out_mae = zinc_held_out_mae
                    best_step = step
                    save_checkpoint(
                        pooler_best_checkpoint_path, step, pooler_without_ddp, pooler_ema,
                        optimizer, lr_scheduler, best_held_out_mae, best_step,
                    )
                    tqdm.write(f"  new best held-out MAE={best_held_out_mae:.4f} @ step {step} -> {pooler_best_checkpoint_path}")
            lib.barrier()

        if is_main:
            append_json_log(
                training_log_path,
                {
                    "step": step, "loss": step_loss_mean.item(), "grad_norm": grad_norm.item(),
                    "lr": lib.deep.get_lr(optimizer),
                    "synthetic_held_out_r2": synthetic_held_out_r2,
                    "zinc_held_out_r2": zinc_held_out_r2,
                    "zinc_held_out_mae": zinc_held_out_mae,
                },
            )

    if is_main:
        print(f"\nbest held-out MAE (real ZINC) = {best_held_out_mae:.4f} at step {best_step}")
        print(f"resume checkpoint: {pooler_checkpoint_path}")
        print(f"best checkpoint:   {pooler_best_checkpoint_path}")
        print(f"per-step log: {training_log_path}")


if __name__ == "__main__":
    main()
