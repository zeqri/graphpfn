"""Extends limix_backbone_gnn_pooling_multi_dataset_varied_prior_fit_test.py
with a more expressive PoolingGNN, aimed at the R2 plateau (~0.30) that
script hit once conv_type diversity was introduced.

Why the base PoolingGNN plateaus: the label-generating SCM
(lib/graphpfn/prior/attributes/layers.py) samples conv_type from
{gcn, sage-mean, sage-min, sage-max, gt} per dataset (with graph_conv_ratio
fixed at 1.0, so every SCM layer is FULLY determined by whichever conv_type
got sampled). Those five split into two structurally different families:
gcn/sage-mean/sage-min/sage-max are order-statistic-style neighbor
REDUCTIONS (sum-normalized, mean, min, max), while gt is content-based
query/key/value ATTENTION. The base PoolingGNN only had a single, fixed
mean-aggregator (dglnn.SAGEConv(aggregator_type="mean")) at both the
per-layer message-passing step AND the final atom->molecule readout -- no
amount of extra depth/width lets a pure mean-aggregator exactly represent
min/max order statistics or attention-weighted combinations, so for datasets
whose label came from sage-min/sage-max/gt, the pooler was structurally the
wrong shape to reconstruct it, regardless of how much it was trained.

This file's PoolingGNN instead gives every layer (and the final readout)
access to mean/min/max reductions AND a lightweight attention branch,
concatenated and linearly projected back down -- see MultiAggregatorConv and
MultiAggregatorPool below. The model doesn't need to know which conv_type
generated a given dataset (that's not observable, and isn't its job to
infer -- inferring the regime from the in-context examples is the FROZEN
backbone's job, via attention over the revealed context labels); its job is
just to not throw away the information (order statistics, attention-weighted
combinations) needed to distinguish those regimes in the first place.

Everything else -- the frozen LimiX-16M backbone, the widened
(min/max-molecules, conv_type-varying) prior, gradient-accumulated
multi-dataset training, prefetching, checkpointing/resume, held-out eval,
DDP design -- is identical to
limix_backbone_gnn_pooling_multi_dataset_varied_prior_fit_test.py; see that
file's docstring for the full rationale behind those pieces.

Usage (single node, 4 GPUs):
    torchrun --nproc_per_node=4 dev/limix_backbone_gnn_pooling_multi_dataset_multi_agg_pooler_fit_test.py \\
        --min-molecules 1000 --max-molecules 4000 --n-sampler-workers 11 --prefetch-factor 4 --n-steps 10000
(Falls back to a single-process, single-GPU/CPU run if launched with plain
`python`, no torchrun -- lib.is_ddp() is False whenever RANK isn't set.)
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

PAPER_DIR = Path(__file__).resolve().parent.parent
if str(PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_DIR))

import dgl  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.parallel import DistributedDataParallel  # noqa: E402
from tqdm import tqdm  # noqa: E402

import lib  # noqa: E402
import lib.deep  # noqa: E402
from lib.graphpfn.prior.attributes import sample_graph_level_labels_via_virtual_node  # noqa: E402
from lib.graphpfn.prior.config import sample_configs  # noqa: E402
from lib.graphpfn.prior.graphs.multi_graph import sample_multi_graph  # noqa: E402
from lib.graphpfn.prior.postprocessing import process_features  # noqa: E402
from lib.graphpfn.prior.prior_typings import unpack  # noqa: E402
from vendor.limix.utils.loading import load_model  # noqa: E402
from dev.limix_encoder_pooling_probe import BASE_PRIOR_CONFIG  # noqa: E402

SEED = 0
CHECKPOINT_PATH = PAPER_DIR / "checkpoints" / "LimiX-16M.ckpt"
TRAIN_FRACTION = 0.8

CONV_TYPES = ["gcn", "sage-mean", "sage-min", "sage-max", "gt"]

N_GNN_LAYERS = 3
N_ATTN_HEADS = 4  # embed_dim=192 (LimiX-16M) / 4 = 48 per head
DROPOUT = 0.0  # matches the real graph-adapter modules' own default (layers.py:
# GraphPFNGraphAttentionModule/GraphPFNMLPModule both default dropout=0.0)

# Gradient-accumulation/ema/lr_scheduler settings taken directly from the
# real pretraining config (pretrain.toml), which already tunes exactly this
# scenario: new adapter parameters trained behind a frozen backbone with a
# fresh synthetic dataset per micro-step. --n-steps defaults to a fraction of
# pretrain.toml's 10000 to keep this a quick dev script; warmup is kept as
# the same 10% RATIO rather than the absolute step count.
N_GRADIENT_ACCUMULATION_STEPS = 20
OPTIMIZER_TYPE = "AdamW"
WEIGHT_DECAY = 0.1
GRADIENT_CLIPPING_NORM = 1.0
LR_SCHEDULER = "cosine"
EMA_DECAY = 0.98

EVAL_EVERY = 20  # optimizer steps between held-out evals
N_EVAL_DATASETS = 8  # matches pretrain.toml's evaluation_data.n_synthetic_per_gpu
EVAL_SEED_BASE = 10_000  # disjoint from every training draw's RNG range

FETCH_STALL_WARN_S = 1.0  # only log a micro-step's fetch time if the prefetch queue ran dry

OUTPUT_ROOT = PAPER_DIR / "dev" / "output" / "limix_backbone_gnn_pooling_multi_dataset_multi_agg_pooler_fit_test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--min-molecules",
        type=int,
        default=300,
        help="Lower bound (inclusive) for molecules/dataset, log_uniform_int-sampled fresh per draw "
        "(default: 300).",
    )
    parser.add_argument(
        "--max-molecules",
        type=int,
        default=3000,
        help="Upper bound (inclusive) for molecules/dataset (default: 3000, BASE_PRIOR_CONFIG's own "
        "value). Bigger datasets cost much more CPU time to sample -- scale --n-sampler-workers up "
        "if you raise this.",
    )
    parser.add_argument(
        "--n-sampler-workers",
        type=int,
        default=6,
        help="Background CPU workers PER RANK that prefetch synthetic datasets (default: 6). Tune "
        "to your job's actual CPU budget: under SLURM, all ranks launched by one `torchrun` share a "
        "single task's --cpus-per-task (NOT --cpus-per-task per rank), so use "
        "roughly (cpus-per-task / nproc_per_node) minus a bit of headroom for each rank's own main "
        "process, not the node's total core count.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="DataLoader prefetch_factor per worker (default: 4) -- how many datasets each worker "
        "queues ahead of time.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=1000,
        help="Total optimizer steps (default: 1000). Also used to size run_output_dir, so changing "
        "this always starts a fresh run rather than resuming/colliding with a different step budget.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Peak learning rate (default: 0.001, same as every earlier fit-test script). Also used "
        "to size run_output_dir, so a different --lr always starts a fresh run rather than resuming "
        "one trained with a different LR schedule.",
    )
    return parser.parse_args()


def build_prior_config(min_molecules: int, max_molecules: int) -> dict:
    """A copy of BASE_PRIOR_CONFIG with two axes widened:
    - n_graphs (molecules/dataset): fixed single value -> log_uniform_int(min_molecules, max_molecules).
    - conv_type: fixed "sage-mean" -> choice(CONV_TYPES), matching pretrain.toml's own 5-option set.
    graph_conv_ratio is left at BASE_PRIOR_CONFIG's fixed 1.0 (every layer fully graph-convolved, so
    conv_type alone determines the aggregation used at every layer). Copied rather than mutated in
    place since limix_encoder_pooling_probe.py (where BASE_PRIOR_CONFIG is defined) and the other
    fit-test scripts both import and rely on the original, fixed version.
    """
    prior_config = copy.deepcopy(BASE_PRIOR_CONFIG)
    prior_values = prior_config["prior"]["values"][0]

    n_graphs_dist = {"_distribution_": "log_uniform_int", "min": min_molecules, "max": max_molecules}
    prior_values["graph"]["sampler"]["values"][0]["n_graphs"] = n_graphs_dist
    prior_values["graph"]["n_nodes"] = n_graphs_dist  # cosmetic only -- the multi-graph sampler
    # (lib/graphpfn/prior/graphs/multi_graph.py) actually sizes datasets from n_graphs/base_n_nodes
    # above, not this top-level field, but keeping it in sync avoids a misleading stale value.

    prior_values["scm"]["conv_type"] = {"_distribution_": "choice", "values": CONV_TYPES}

    return prior_config


def build_run_paths(min_molecules: int, max_molecules: int, n_steps: int, lr: float) -> tuple[Path, Path, Path]:
    """run_output_dir encodes --min-molecules/--max-molecules/--n-steps/--lr so different configs get
    separate checkpoints/logs -- otherwise re-running with different bounds/LR would try to
    auto-resume a checkpoint trained (and already fully stepped, per the OLD --n-steps) on a totally
    different regime, silently doing zero further training.
    """
    run_output_dir = OUTPUT_ROOT / f"molecules_{min_molecules}_{max_molecules}_n_steps_{n_steps}_lr_{lr:g}"
    return run_output_dir, run_output_dir / "pooler_checkpoint.pt", run_output_dir / "training_log.jsonl"


class MultiAggregatorConv(nn.Module):
    """Per-layer message passing combining mean/min/max neighbor reductions
    with a lightweight single/multi-head dot-product attention aggregation,
    concatenated and projected back to d_out.

    Targets the 5 conv_types the label-generating SCM can use (gcn/
    sage-mean/sage-min/sage-max/gt, see this file's module docstring): a
    single fixed mean-aggregator can't represent min/max order statistics or
    content-based attention no matter how much depth/width it's given, so
    this layer gives every message-passing step the raw ingredients for all
    five regimes instead, letting the (frozen, downstream) backbone's
    in-context attention pick out which combination fits a given dataset.
    """

    def __init__(self, d_in: int, d_out: int, n_heads: int = 1):
        super().__init__()
        assert d_out % n_heads == 0, f"d_out={d_out} must be divisible by n_heads={n_heads}"
        self.d_out = d_out
        self.n_heads = n_heads
        self.d_head = d_out // n_heads
        self.attn_scale = self.d_head**-0.5

        self.pre_linear = nn.Linear(d_in, d_out)  # shared projection feeding every aggregation branch
        self.attn_qkv = nn.Linear(d_out, d_out * 3)
        self.attn_out = nn.Linear(d_out, d_out)
        # own (self) transform + mean + min + max + attention -> project back to d_out
        self.combine = nn.Linear(d_out * 5, d_out)

    def forward(self, graph: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        h = self.pre_linear(x)

        mean_msg = dgl.ops.copy_u_mean(graph, h)
        min_msg = dgl.ops.copy_u_min(graph, h)
        max_msg = dgl.ops.copy_u_max(graph, h)

        # Self-loops so every node attends to (at least) itself, mirroring
        # lib/graphpfn/prior/attributes/layers.py's GTConv exactly.
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
    """atom_out: (n_atoms, embed_dim) -> (n_molecules, embed_dim) via
    torch's scatter_reduce (stable since torch 2.0). include_self=False so
    `init_value` never contaminates the result -- safe since every molecule
    has at least one atom, so every output row gets at least one real value
    reduced into it.
    """
    n_atoms, embed_dim = atom_out.shape
    out = torch.full((n_molecules, embed_dim), init_value, device=atom_out.device, dtype=atom_out.dtype)
    index = molecule_id.unsqueeze(-1).expand(-1, embed_dim)
    return out.scatter_reduce(0, index, atom_out, reduce=reduce, include_self=False)


class MultiAggregatorPool(nn.Module):
    """Atom->molecule readout combining mean/min/max pooling (concatenated +
    projected) instead of a single fixed mean -- same rationale as
    MultiAggregatorConv, applied to the final readout step instead of the
    per-layer message passing (the base PoolingGNN's final pooling step was
    ALSO hardcoded to a plain mean, same mismatch).
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.combine = nn.Linear(embed_dim * 3, embed_dim)

    def forward(self, atom_out: torch.Tensor, molecule_id: torch.Tensor, n_molecules: int) -> torch.Tensor:
        mean_pool = _scatter_pool(atom_out, molecule_id, n_molecules, reduce="mean", init_value=0.0)
        min_pool = _scatter_pool(atom_out, molecule_id, n_molecules, reduce="amin", init_value=float("inf"))
        max_pool = _scatter_pool(atom_out, molecule_id, n_molecules, reduce="amax", init_value=float("-inf"))
        combined = torch.cat([mean_pool, min_pool, max_pool], dim=-1)
        return self.combine(combined)


class PoolingGNN(nn.Module):
    """Message passing (real bonds) + atom->molecule pooling, applied
    independently per feature-group with SHARED weights across groups (the
    same MultiAggregatorConv/LayerNorm/MultiAggregatorPool instances are
    called once per group, looping over the group dim) -- matches how the
    real graph adapter (GraphPFNGraphAttentionModule) natively operates on
    the (n_nodes, n_groups, embed_dim) shape, instead of collapsing groups to
    a single vector before message passing.

    See this file's module docstring and MultiAggregatorConv/
    MultiAggregatorPool's docstrings for why both the per-layer aggregation
    AND the final readout now combine mean/min/max/attention instead of a
    single fixed mean.
    """

    def __init__(self, embed_dim: int, n_layers: int, dropout: float, n_heads: int = 1):
        super().__init__()
        self.convs = nn.ModuleList(
            MultiAggregatorConv(embed_dim, embed_dim, n_heads=n_heads) for _ in range(n_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(embed_dim) for _ in range(n_layers))
        self.dropout = nn.Dropout(dropout)
        self.pool = MultiAggregatorPool(embed_dim)

    def forward(
        self,
        graph: dgl.DGLGraph,
        atom_embeddings_grouped: torch.Tensor,
        molecule_id: torch.Tensor,
        n_molecules: int,
    ) -> torch.Tensor:
        n_atoms, n_groups, embed_dim = atom_embeddings_grouped.shape
        dtype = atom_embeddings_grouped.dtype

        pooled_per_group = []
        for g in range(n_groups):
            h = atom_embeddings_grouped[:, g, :].float()  # graph ops want float32
            for conv, norm in zip(self.convs, self.norms):
                h = self.dropout(F.relu(norm(conv(graph, h))))
            pooled_per_group.append(self.pool(h, molecule_id, n_molecules))
        pooled = torch.stack(pooled_per_group, dim=1)  # (n_molecules, n_groups, embed_dim)
        return pooled.to(dtype)


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def save_checkpoint(
    path: Path,
    step: int,
    pooler_without_ddp: nn.Module,
    pooler_ema: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    best_held_out_r2: float,
    best_step: int,
) -> None:
    """Everything needed to resume this run from `step` onward with the SAME
    config (see build_run_paths' docstring for why this can't extend the
    step budget). Written to a .tmp path and renamed into place so a crash
    mid-write never leaves a corrupt checkpoint at `path`.
    """
    tmp_path = path.with_suffix(".tmp")
    torch.save(
        {
            "step": step,
            "pooler": pooler_without_ddp.state_dict(),
            "pooler_ema": pooler_ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "best_held_out_r2": best_held_out_r2,
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


@dataclass
class SampledDataset:
    graph: dgl.DGLGraph
    atom_embeddings_grouped: torch.Tensor  # (n_atoms, n_groups, embed_dim), frozen encoder_x output
    molecule_id: torch.Tensor
    n_molecules: int
    y_norm: torch.Tensor  # (n_molecules,), standardized against this dataset's own train split
    eval_pos_molecules: int  # first `eval_pos_molecules` indices (post-reorder) are context/train
    conv_type: str  # which of CONV_TYPES generated this dataset's label -- for the per-conv_type
    # held-out R2 breakdown (evaluate_held_out), so a low average can be told apart from "uniformly
    # mediocre" vs. "some conv_types (e.g. gt) still poorly fit while others are already strong."


def _sample_prior_dataset_with_conv_type(prior_config: dict) -> tuple[torch.Tensor, dgl.DGLGraph, torch.Tensor, str]:
    """Same body as lib/graphpfn/prior/graph_level.py's sample_graph_level_dataset,
    but also returns the resolved conv_type for this draw. The public
    function doesn't expose it, and calling sample_configs a second time to
    recover it would consume RNG state again and desync from the dataset
    that was actually generated -- so the whole body is duplicated here
    instead of wrapping the original.
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
    """The CPU-bound, model-free half of dataset prep: sample one synthetic
    molecule dataset from the prior, reorder it context-first, and pad
    features to a features_per_group multiple. No GPU/model access at all --
    this is exactly what's safe to run inside a DataLoader worker
    subprocess, in parallel, while the main process is busy on the GPU.

    Returns a plain dict of CPU tensors, NOT a SampledDataset -- the graph is
    passed as raw (src, dst) edge tensors rather than a dgl.DGLGraph, since
    reconstructing it with dgl.graph(...) in the main process (see
    encode_raw_dataset_on_gpu) is simpler and cheaper than relying on
    DGLGraph objects surviving multiprocessing IPC/pickling.
    """
    atom_features, graph, y_per_molecule, conv_type = _sample_prior_dataset_with_conv_type(prior_config)

    # Context-first molecule reorder (unbatch/rebatch keeps each molecule's
    # own atoms + real bonds + features together, correctly, without manual
    # edge-index math).
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
        "atom_features": atom_features_reordered,  # (n_atoms, n_features_padded)
        "edges_src": src,
        "edges_dst": dst,
        "n_atoms": n_atoms,
        "molecule_id": molecule_id,  # (n_atoms,)
        "n_molecules": n_molecules,
        "y_norm": y_norm,  # (n_molecules,)
        "eval_pos_atoms": eval_pos_atoms,
        "eval_pos_molecules": n_train,
        "conv_type": conv_type,
    }


class _RawSyntheticDatasetIterable(torch.utils.data.IterableDataset):
    """Infinite stream of raw (CPU-only) synthetic datasets, fed to a
    background-worker DataLoader (see build_prefetch_loader) so sampling for
    micro-step i+1 overlaps with the GPU work (encode/forward/backward) of
    micro-step i, instead of blocking the training loop like a synchronous
    sample_and_prepare_dataset call would.
    """

    def __init__(self, seed: int, features_per_group: int, prior_config: dict):
        self.seed = seed
        self.features_per_group = features_per_group
        self.prior_config = prior_config

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        # sample_graph_level_dataset reads the GLOBAL torch/numpy RNG, so
        # each worker process needs its own non-colliding seed -- otherwise
        # every worker would (re-)sample the identical sequence of datasets.
        torch.manual_seed(self.seed + worker_id)
        np.random.seed(self.seed + worker_id)
        while True:
            yield _sample_raw_dataset(self.features_per_group, self.prior_config)


def build_prefetch_loader(
    seed: int,
    features_per_group: int,
    prior_config: dict,
    n_workers: int,
    prefetch_factor: int,
):
    """A DataLoader with background worker processes doing the CPU-bound
    sampling in parallel and ahead of time, queued up via prefetch_factor --
    mirrors bin/graphpfn/sampler.py's GraphPriorSampler (n_workers,
    prefetch_factor), just without its batch-padding-across-datasets
    machinery, since we still process one dataset per micro-step here.
    """
    dataset = _RawSyntheticDatasetIterable(seed=seed, features_per_group=features_per_group, prior_config=prior_config)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=n_workers,
        prefetch_factor=prefetch_factor if n_workers > 0 else None,
        persistent_workers=n_workers > 0,
        collate_fn=lambda batch: batch[0],
    )
    return iter(loader)


def encode_raw_dataset_on_gpu(model: nn.Module, device: torch.device, raw: dict) -> SampledDataset:
    """The GPU half of dataset prep: rebuild the dgl graph from edge-index
    tensors (cheap) and run the frozen x_preprocess -> process_4_x ->
    encoder_x chain (matches FeaturesTransformer.forward's exact order,
    called manually since our "sequence" for this stage is ATOMS, not
    molecules). This is the only part of dataset prep that needs the
    GPU-resident model, so it's the only part that can't run in a
    background worker.
    """
    n_atoms = raw["n_atoms"]
    n_molecules = raw["n_molecules"]
    graph = dgl.graph((raw["edges_src"], raw["edges_dst"]), num_nodes=n_atoms).to(device)
    atom_features = raw["atom_features"].to(device)
    molecule_id = raw["molecule_id"].to(device)
    y_norm = raw["y_norm"].to(device)
    eval_pos_atoms = raw["eval_pos_atoms"]

    features_per_group = model.features_per_group
    n_groups = atom_features.shape[-1] // features_per_group

    x = atom_features.unsqueeze(0)  # (1, n_atoms, n_features_padded)
    x_dict = {"data": x, "mask": torch.isnan(x).to(torch.int32)}
    x_dict = {k: v.reshape(1, n_atoms, n_groups, features_per_group) for k, v in x_dict.items()}
    x_dict["eval_pos"] = eval_pos_atoms
    with torch.no_grad():
        preprocessed = model.x_preprocess(x_dict)
        preprocessed = model.process_4_x(preprocessed)
        x_encoder_result = model.encoder_x(preprocessed)
    atom_embeddings_grouped = x_encoder_result["data"].squeeze(0)  # (n_atoms, n_groups, embed_dim)

    return SampledDataset(
        graph=graph,
        atom_embeddings_grouped=atom_embeddings_grouped,
        molecule_id=molecule_id,
        n_molecules=n_molecules,
        y_norm=y_norm,
        eval_pos_molecules=raw["eval_pos_molecules"],
        conv_type=raw["conv_type"],
    )


def sample_and_prepare_dataset(model: nn.Module, device: torch.device, prior_config: dict) -> SampledDataset:
    """Synchronous (non-prefetched) dataset sampling -- used only for the
    fixed-seed held-out eval datasets, which are sampled just once every
    EVAL_EVERY steps, so background prefetching isn't worth the complexity
    there. Training uses the prefetched path (build_prefetch_loader +
    encode_raw_dataset_on_gpu) instead.
    """
    raw = _sample_raw_dataset(model.features_per_group, prior_config)
    return encode_raw_dataset_on_gpu(model, device, raw)


@contextmanager
def _temporary_rng_seed(seed: int):
    """Saves/restores the torch+numpy global RNG state so sampling a
    deterministic held-out eval dataset never perturbs the training draw
    sequence (mirrors bin/graphpfn/pretrain.py's
    `@delu.random.preserve_state()` on get_synthetic_eval_dataset).
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


def sample_eval_dataset(model: nn.Module, device: torch.device, idx: int, prior_config: dict) -> SampledDataset:
    """A fixed-seed held-out dataset: the same `idx` always resamples the
    exact same molecules/labels, and its seed range (EVAL_SEED_BASE+) never
    overlaps any unseeded training draw, so it never leaks into training.
    """
    with _temporary_rng_seed(EVAL_SEED_BASE + idx):
        return sample_and_prepare_dataset(model, device, prior_config)


def forward_pass(model: nn.Module, dataset: SampledDataset, pooler_module: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Runs the frozen backbone's add_embeddings -> mixed_y_embedding ->
    transformer_encoder -> y_decoder chain over ALL of `dataset`'s molecules
    (context labels revealed, query masked to NaN). Returns
    (query_predictions, query_targets).
    """
    eval_pos = dataset.eval_pos_molecules
    pooled_grouped = pooler_module(dataset.graph, dataset.atom_embeddings_grouped, dataset.molecule_id, dataset.n_molecules)
    pooled_x = pooled_grouped.unsqueeze(0).to(next(model.encoder_x.parameters()).dtype)

    embedded_x = model.add_embeddings(pooled_x)  # (1, n_molecules, n_groups, embed_dim)

    y_local = dataset.y_norm.unsqueeze(0).unsqueeze(-1).clone()  # (1, n_molecules, 1)
    y_dict = {"data": y_local}
    y_dict["data"][:, eval_pos:] = torch.nan
    y_type = torch.ones_like(y_dict["data"])  # all regression
    embedded_y = model.mixed_y_embedding(y_dict, y_type=y_type, eval_pos=eval_pos)

    embedded_all = torch.cat((embedded_x, embedded_y.unsqueeze(2).to(embedded_x.dtype)), dim=2)
    encoder_out = model.transformer_encoder(embedded_all, feature_atten_mask=None, eval_pos=eval_pos)[0]
    encoder_out = model.encoder_out_norm(encoder_out)

    test_encoder_out = encoder_out[:, eval_pos:, -1]
    test_y_type = y_type[:, eval_pos:]
    _, reg_output = model.y_decoder(test_encoder_out, test_y_type)
    pred = reg_output.float().squeeze(0).squeeze(-1)  # (n_query,)
    target = dataset.y_norm[eval_pos:]
    return pred, target


def main() -> None:
    args = parse_args()
    n_warmup_steps = max(1, round(args.n_steps * 0.1))
    prior_config = build_prior_config(args.min_molecules, args.max_molecules)
    run_output_dir, pooler_checkpoint_path, training_log_path = build_run_paths(
        args.min_molecules, args.max_molecules, args.n_steps, args.lr
    )

    if lib.is_ddp():
        lib.configure_ddp()
    rank = lib.get_rank()
    world_size = lib.get_world_size()
    device = lib.get_device()
    is_main = lib.is_master_process()

    if is_main:
        run_output_dir.mkdir(parents=True, exist_ok=True)
    lib.barrier()  # other ranks wait so a resume check below never races directory creation

    # Unlike a single-dataset script, every rank must draw DIFFERENT training
    # datasets (that's the point of data-parallel training here), so each
    # rank gets its own seed.
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
            f"PoolingGNN: MultiAggregatorConv (mean/min/max/attention, {N_ATTN_HEADS} heads) x "
            f"{N_GNN_LAYERS} layers + MultiAggregatorPool readout"
        )
        print(
            f"Training on a FRESH synthetic dataset every micro-step "
            f"({args.min_molecules}-{args.max_molecules} molecules, log-uniform per draw; "
            f"conv_type in {CONV_TYPES}, graph_conv_ratio=1.0 fixed; causal.enabled=False fixed) -- "
            f"lr={args.lr}, {N_GRADIENT_ACCUMULATION_STEPS} per optimizer step, {args.n_steps} steps "
            f"total = {args.n_steps * N_GRADIENT_ACCUMULATION_STEPS} datasets), evaluating on "
            f"{N_EVAL_DATASETS} fixed-seed held-out datasets every {EVAL_EVERY} steps..."
        )
        print(
            f"Prefetching training datasets with {args.n_sampler_workers} background workers/rank "
            f"(prefetch_factor={args.prefetch_factor})..."
        )
        print(f"Run output dir: {run_output_dir}")

    # Each rank needs its own non-colliding worker seed range (separate from
    # the SEED + rank used above for this process's own RNG, and from
    # EVAL_SEED_BASE) -- see _RawSyntheticDatasetIterable.__iter__.
    raw_dataset_iter = build_prefetch_loader(
        seed=SEED + rank * 100_000,
        features_per_group=model.features_per_group,
        prior_config=prior_config,
        n_workers=args.n_sampler_workers,
        prefetch_factor=args.prefetch_factor,
    )

    pooler_without_ddp = PoolingGNN(
        embed_dim=embed_dim, n_layers=N_GNN_LAYERS, dropout=DROPOUT, n_heads=N_ATTN_HEADS
    ).to(device)
    pooler = pooler_without_ddp
    if lib.is_ddp():
        # device_ids is only valid for GPU modules; CPU (or CPU-fallback,
        # e.g. no CUDA available) DDP must omit it entirely.
        ddp_device_ids = [lib.get_local_rank()] if device.type == "cuda" else None
        pooler = DistributedDataParallel(pooler_without_ddp, device_ids=ddp_device_ids)

    params = lib.deep.make_parameter_groups(pooler_without_ddp)
    optimizer = lib.deep.make_optimizer(type=OPTIMIZER_TYPE, lr=args.lr, weight_decay=WEIGHT_DECAY, params=params)
    lr_scheduler = lib.deep.get_lr_scheduler(
        optimizer, n_warmup_steps=n_warmup_steps, n_steps=args.n_steps, scheduler=LR_SCHEDULER
    )
    ema_multi_avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(decay=EMA_DECAY)
    pooler_ema = torch.optim.swa_utils.AveragedModel(pooler_without_ddp, device, multi_avg_fn=ema_multi_avg_fn)

    # Resume support: every rank independently reads the same checkpoint file
    # (safe -- it's read-only from here, and all ranks need identical model/
    # optimizer/scheduler state). Only valid for continuing an INTERRUPTED
    # run of this SAME config; see build_run_paths' docstring for why this
    # can't extend n_steps (that always gets a fresh output dir instead).
    start_step = 1
    best_held_out_r2 = -float("inf")
    best_step = -1
    if pooler_checkpoint_path.exists():
        checkpoint = load_checkpoint(pooler_checkpoint_path, device)
        pooler_without_ddp.load_state_dict(checkpoint["pooler"])
        pooler_ema.load_state_dict(checkpoint["pooler_ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        best_held_out_r2 = checkpoint["best_held_out_r2"]
        best_step = checkpoint["best_step"]
        start_step = checkpoint["step"] + 1
        if is_main:
            print(
                f"Resumed from {pooler_checkpoint_path} at step {checkpoint['step']} "
                f"(best held-out R2 so far: {best_held_out_r2:.4f} @ step {best_step})"
            )

    def evaluate_held_out() -> float:
        """Mean R2 over N_EVAL_DATASETS fixed-seed held-out draws, using the
        EMA weights -- genuinely unseen data (a disjoint seed range from
        every training draw), so it's an actual generalization metric. Also
        prints each dataset's own conv_type + n_molecules + R2 (not just the
        average), so a low average can be told apart from "uniformly
        mediocre" vs. "some conv_types (e.g. gt) still poorly fit while
        others are already strong" -- see SampledDataset.conv_type's
        docstring. n_molecules is included specifically to check whether
        min/max-aggregated datasets stall out more on LARGER molecule counts
        (the min/max of more atoms is a more extreme, less informative
        statistic than the mean, regardless of pooler architecture -- an
        info-theoretic property of the label, not necessarily a fixable
        model limitation). Only ever called from within an `if is_main:`
        block.
        """
        pooler_ema.eval()
        per_dataset = []
        with torch.no_grad():
            for idx in range(N_EVAL_DATASETS):
                dataset = sample_eval_dataset(model, device, idx, prior_config)
                pred, target = forward_pass(model, dataset, pooler_ema)
                per_dataset.append((dataset.conv_type, dataset.n_molecules, r2_score(pred, target)))
        breakdown = " | ".join(f"{conv_type}(n={n_molecules})={r2:.3f}" for conv_type, n_molecules, r2 in per_dataset)
        tqdm.write(f"  held-out per-dataset: {breakdown}")
        return float(np.mean([r2 for _, _, r2 in per_dataset]))

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
                # Pulls a dataset a background worker already finished
                # sampling (see build_prefetch_loader) -- ideally near-instant.
                raw = next(raw_dataset_iter)
                t1 = time.time()
                dataset = encode_raw_dataset_on_gpu(model, device, raw)
                t2 = time.time()
                pred, target = forward_pass(model, dataset, pooler)
                loss = F.mse_loss(pred, target)
                t3 = time.time()
                (loss / N_GRADIENT_ACCUMULATION_STEPS).backward()
                t4 = time.time()
                step_losses.append(loss.detach())
            fetch_time = t1 - t0
            if is_main and fetch_time > FETCH_STALL_WARN_S:
                # Only prints when the prefetch queue actually ran dry (a
                # worker didn't keep up) -- with prefetching working
                # normally, fetch is near-instant and not worth a line every
                # micro-step (that was drowning out the per-step loss below).
                total_microsteps_done = (step - 1) * N_GRADIENT_ACCUMULATION_STEPS + inner_step + 1
                tqdm.write(
                    f"[stall] micro-step {total_microsteps_done} (step {step}, "
                    f"{inner_step + 1}/{N_GRADIENT_ACCUMULATION_STEPS}) waited fetch={fetch_time:.2f}s "
                    f"for the prefetch queue -- workers aren't keeping up; consider raising "
                    f"--n-sampler-workers/--prefetch-factor."
                )

        grad_norm = torch.nn.utils.clip_grad_norm_(pooler.parameters(), GRADIENT_CLIPPING_NORM)
        optimizer.step()
        pooler_ema.update_parameters(pooler_without_ddp)
        lr_scheduler.step()
        step_loss_mean = torch.stack(step_losses).mean()

        if is_main:
            if isinstance(iterator, tqdm):
                iterator.set_postfix(loss=step_loss_mean.item(), grad_norm=grad_norm.item(), lr=lib.deep.get_lr(optimizer))
            # A durable, scrollback-friendly line every step -- unlike the
            # tqdm postfix above (which just redraws the same line in place
            # and is easy to lose track of), this one persists in the log.
            tqdm.write(
                f"step {step:5d} | loss {step_loss_mean.item():.4f} | grad_norm {grad_norm.item():.4f} | "
                f"lr {lib.deep.get_lr(optimizer):.6f}"
            )

        held_out_r2 = None
        if step % EVAL_EVERY == 0 or step == args.n_steps:
            if is_main:
                held_out_r2 = evaluate_held_out()
                if held_out_r2 > best_held_out_r2:
                    best_held_out_r2 = held_out_r2
                    best_step = step
                tqdm.write(
                    f"step {step:5d} | held-out R2 (EMA, {N_EVAL_DATASETS} datasets) = {held_out_r2:.4f}"
                )
                save_checkpoint(
                    pooler_checkpoint_path,
                    step,
                    pooler_without_ddp,
                    pooler_ema,
                    optimizer,
                    lr_scheduler,
                    best_held_out_r2,
                    best_step,
                )
            lib.barrier()

        if is_main:
            append_json_log(
                training_log_path,
                {
                    "step": step,
                    "loss": step_loss_mean.item(),
                    "grad_norm": grad_norm.item(),
                    "lr": lib.deep.get_lr(optimizer),
                    "held_out_r2": held_out_r2,
                },
            )

    if is_main:
        print(f"\nbest held-out R2 = {best_held_out_r2:.4f} at step {best_step}")
        print(f"checkpoint: {pooler_checkpoint_path}")
        print(f"per-step log: {training_log_path}")


if __name__ == "__main__":
    main()
