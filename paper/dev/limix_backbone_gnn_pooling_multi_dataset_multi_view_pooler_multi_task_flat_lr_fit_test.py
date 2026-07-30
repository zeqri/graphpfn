"""Extends limix_backbone_gnn_pooling_multi_dataset_multi_view_pooler_fit_test.py
to widen the TASK dimension: instead of training/evaluating on regression
only, this script samples a task type (regression/binclass/multiclass) per
dataset, matching how the real pretraining pipeline
(exp/graphpfn/pretrain/main/pretrain.toml, run via bin/graphpfn/pretrain.py)
trains across all three.

Why this wasn't already covered: sample_graph_level_labels_via_virtual_node
(the graph-level pathway this whole family of scripts is built on) only
ever emits a raw continuous per-molecule label. Unlike the real node-level
pretraining pipeline (lib/graphpfn/prior/priors/graph_then_attributes.py),
the graph-level pathway never calls
apply_task (lib/graphpfn/prior/postprocessing/task.py) itself -- so every
earlier fit-test script in this family trained purely on regression
(y_type hardcoded to "all regression"), regardless of what pretrain.toml's
own task._type_ distribution looks like. This script:
  1. Adds a "task" config to the prior (mirroring pretrain.toml's
     [base_config.prior.prior.values.task] block: _type_ choice among
     regression/binclass/multiclass, quantile, n_classes, multiclass_type,
     p_ordered, p_reverse) -- see build_prior_config.
  2. Calls apply_task manually right after the raw continuous label is
     drawn, since the graph-level pathway doesn't do this on its own -- see
     _sample_prior_dataset_with_conv_type_and_task.
  3. Sets y_type (vendor/limix/model/transformer.py's
     mixed_y_embedding/y_decoder convention: 0=classification branch,
     1=regression branch) per-dataset from the sampled task_type, instead
     of hardcoding "all regression" -- see forward_pass.
  4. Uses cross-entropy (over the model's fixed MAX_N_CLASSES-wide
     cls_output) for binclass/multiclass, MSE (over reg_output) for
     regression -- matching bin/graphpfn/pretrain.py's loss branch exactly
     (both classification task types share the same cross_entropy call;
     binclass is just 2-class cross-entropy) -- see compute_loss.
  5. Reports R2 for regression, accuracy for multiclass, and average
     precision for binclass as the held-out metric -- matching
     lib/graph/data.py's get_score per-task-type choice -- see
     _task_metric.
Since apply_task's own class-coverage guarantee
(lib/graphpfn/prior/checks.py's check_dataset) is keyed to a train/test
split this script doesn't have yet at the point apply_task runs (the
context-first molecule reorder happens afterward, in
_sample_raw_dataset), the classification-only sanity checks
(check_n_classes/check_class_coverage) are re-run here, post-reorder, with
a small resample-from-scratch retry budget (_check_label_sanity,
N_LABEL_SANITY_RETRIES) -- the one new dataset-rejection mode task-type
sampling introduces (regression datasets never trigger it).

Real-data ICL eval (ogbg-molhiv): alongside the synthetic held-out eval
(evaluate_held_out, purely synthetic-prior datasets), this script also
tracks in-context learning on a REAL molecular benchmark as training
progresses -- a fixed, seeded random 2000-graph sample of ogbg-molhiv's
train split as the revealed in-context "demonstrations", and the full
(filtered, see below) test split as the masked query -- see
load_molhiv_context_and_query/_molhiv_graphs_to_raw_dataset/
evaluate_molhiv. This reuses the exact same encode_raw_dataset_on_gpu/
forward_pass pipeline as the synthetic path (ogbg-molhiv's 9-dim integer
atom features are just padded to a features_per_group multiple and passed
through as plain floats -- NOT run through process_features, since that
pipeline's categorical/permutation logic is specific to the synthetic
prior's own feature semantics; the frozen backbone's own x_preprocess/
encoder_x already normalizes internally). ogbg-molhiv is binary
classification (HIV inhibition), so task_type is fixed to
TaskType.BINCLASS; the logged metric is ROC-AUC (sklearn.metrics.
roc_auc_score) -- OGB's own metric for this dataset -- rather than the
average-precision used for synthetic binclass draws (_task_metric),
matching what's directly comparable to published ogbg-molhiv results. Runs
on the same EVAL_EVERY cadence as evaluate_held_out, main-process only; the
context is sampled ONCE at the start of training (not re-sampled per eval
call) so successive calls track improvement on the exact same slice.
Both the context pool and the query set first drop any graph containing an
isolated atom (~5.5% of ogbg-molhiv, mostly salts with a disconnected
counter-ion, e.g. Na+/Cl-) -- the synthetic prior never produces
disconnected fragments, so PoolingGNN's mean/min/max message-passing
aggregation was never built to handle a zero-neighbor node (it reduces to
NaN/inf there, which then poisons every OTHER molecule's prediction too via
the frozen backbone's cross-molecule attention); see
_graph_has_isolated_atom.

Everything else -- the frozen LimiX-16M backbone, PoolingGNN/MultiViewPool
(the atom<->atom message-passing + atom->molecule readout architecture),
the widened (min/max-molecules, conv_type-varying) prior, gradient-
accumulated multi-dataset training, prefetching, checkpointing/resume,
per-conv_type breakdown, DDP design -- is identical to
limix_backbone_gnn_pooling_multi_dataset_multi_view_pooler_fit_test.py; see
that file's docstring (and, further back,
limix_backbone_gnn_pooling_multi_dataset_multi_agg_pooler_fit_test.py's)
for the full rationale behind those pieces.

Two changes vs.
limix_backbone_gnn_pooling_multi_dataset_multi_view_pooler_multi_task_fit_test.py
(the direct predecessor of this file), both validated on a single-fixed-
dataset binclass sanity check
(dev/limix_backbone_gnn_pooling_binclass_fit_test.py) before porting here:
  1. No LR schedule at all -- --lr is now a FLAT, constant learning rate
     (set once on the optimizer, never touched again), not the peak of a
     cosine-with-warmup schedule. The predecessor script's own training logs
     showed loss/grad_norm freezing near chance level regardless of peak LR
     (0.001 collapsed by step ~100, 0.0005 only delayed it) -- the common
     factor was RAMPING UP toward a peak, not the peak's specific value.
     The single-dataset sanity check confirmed: a flat LR from step 1 (no
     warmup, no decay) let the same architecture reach a PERFECT held-out
     fit, where every ramping schedule tried had collapsed. See compute_loss
     and main() for where the cosine scheduler/n_warmup_steps machinery was
     removed.
  2. LABEL_SMOOTHING on the classification cross-entropy loss -- caps how
     confident the target is allowed to be (e.g. 0.9 instead of 1.0),
     directly addressing WHY ramping schedules kept collapsing: standard
     cross-entropy's gradient vanishes as predicted probabilities approach
     0/1 (right or wrong), which is exactly the saturation trap a rising LR
     kept walking the optimizer into. Kept even with the flat LR fix, since
     both changes were validated together on the single-dataset check and
     either one alone might not be sufficient here (this script's task
     diversity -- 5 conv_types x wide quantile-driven class imbalance -- is
     harder than that check's deliberately-simplified single balanced
     dataset).
Also added: sys.stdout.reconfigure(line_buffering=True) at the top of
main() -- tqdm.write()'s eval-line prints were observed sitting in a stdout
buffer and not reaching the SLURM .out file until much later (the progress
bar's own redraws flush via a different, more eager path), which made
training look like it had stalled with no eval output for many steps when
it hadn't. This is unrelated to the LR/loss changes but was found and fixed
alongside them.

This file is a fully separate script (not an edit to the predecessor) so
that script's already-running training job/checkpoint under its own
run_output_dir is left untouched.

Usage (single node, 4 GPUs):
    torchrun --nproc_per_node=4 dev/limix_backbone_gnn_pooling_multi_dataset_multi_view_pooler_multi_task_flat_lr_fit_test.py \\
        --min-molecules 1000 --max-molecules 3000 --n-sampler-workers 11 --prefetch-factor 4 --n-steps 10000 --lr 0.0001
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
import sklearn.metrics  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from ogb.graphproppred import PygGraphPropPredDataset  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch.nn.parallel import DistributedDataParallel  # noqa: E402
from tqdm import tqdm  # noqa: E402

import lib  # noqa: E402
import lib.deep  # noqa: E402
from lib.graphpfn.prior.attributes import sample_graph_level_labels_via_virtual_node  # noqa: E402
from lib.graphpfn.prior.checks import SanityCheckError, check_class_coverage, check_n_classes  # noqa: E402
from lib.graphpfn.prior.config import sample_configs  # noqa: E402
from lib.graphpfn.prior.graphs.multi_graph import sample_multi_graph  # noqa: E402
from lib.graphpfn.prior.postprocessing import apply_task, process_features  # noqa: E402
from lib.graphpfn.prior.prior_typings import unpack  # noqa: E402
from lib.util import TaskType  # noqa: E402
from vendor.limix.utils.loading import load_model  # noqa: E402
from dev.limix_encoder_pooling_probe import BASE_PRIOR_CONFIG  # noqa: E402

SEED = 0
CHECKPOINT_PATH = PAPER_DIR / "checkpoints" / "LimiX-16M.ckpt"
TRAIN_FRACTION = 0.8

CONV_TYPES = ["gcn", "sage-mean", "sage-min", "sage-max", "gt"]

# CONV_TYPES = ["gcn", "sage-mean"]
# TASK_TYPES = ["regression", "binclass", "multiclass"]  # matches pretrain.toml's task._type_ choice
TASK_TYPES = [ "binclass"]  # matches pretrain.toml's task._type_ choice
MAX_N_CLASSES = 10  # matches pretrain.toml's task.n_classes max AND the LimiX-16M checkpoint's
# fixed decoder_config['num_classes'] width (asserted at runtime in main(), right after the
# checkpoint loads, against model.decoder_config) -- raising this beyond what cls_y_decoder/
# cls_y_encoder were actually sized for would silently misbehave (out-of-range class ids) rather
# than error.
N_LABEL_SANITY_RETRIES = 5  # resample-from-scratch budget for a failed class-coverage/n_classes
# check (see _check_label_sanity) -- the one new dataset-rejection mode task-type sampling
# introduces beyond what the graph/SCM sampling machinery already retries internally on its own.

N_GNN_LAYERS = 3
N_ATTN_HEADS = 4  # embed_dim=192 (LimiX-16M) / 4 = 48 per head -- message-passing attention branch only
N_POOL_VIEWS = 4  # mean, min, max, learned-attention -- see MultiViewPool
DROPOUT = 0.0  # matches the real graph-adapter modules' own default (layers.py:
# GraphPFNGraphAttentionModule/GraphPFNMLPModule both default dropout=0.0)

# Gradient-accumulation/ema settings taken directly from the real pretraining config
# (pretrain.toml), which already tunes exactly this scenario: new adapter parameters
# trained behind a frozen backbone with a fresh synthetic dataset per micro-step.
# --n-steps defaults to a fraction of pretrain.toml's 10000 to keep this a quick dev
# script. NOTE: unlike pretrain.toml (and every earlier script in this family), there is
# NO lr_scheduler/warmup here -- --lr is a flat, constant rate -- see module docstring.
N_GRADIENT_ACCUMULATION_STEPS = 20
OPTIMIZER_TYPE = "AdamW"
WEIGHT_DECAY = 0.1
GRADIENT_CLIPPING_NORM = 1.0
EMA_DECAY = 0.98

# See module docstring's point 2 -- caps how confident the classification loss's target
# is allowed to be, keeping cross-entropy's gradient from ever fully saturating.
LABEL_SMOOTHING = 0.1

EVAL_EVERY = 20  # optimizer steps between held-out evals
N_EVAL_DATASETS = 8  # matches pretrain.toml's evaluation_data.n_synthetic_per_gpu
EVAL_SEED_BASE = 10_000  # disjoint from every training draw's RNG range

FETCH_STALL_WARN_S = 1.0  # only log a micro-step's fetch time if the prefetch queue ran dry

MOLHIV_ROOT = Path("/p/project1/profound/al-zeqri1/graphs/ogb/dataset")  # PygGraphPropPredDataset's
# `root` -- it looks for root/ogbg_molhiv itself, already processed there (no download needed).
MOLHIV_N_CONTEXT = 2000  # size of the fixed, seeded random in-context "demonstration" sample drawn
# from ogbg-molhiv's train split (32901 graphs) -- the query is always the FULL test split (~4113
# graphs), never a subset.
MOLHIV_EVAL_SEED = 20_000  # disjoint from SEED/rank, EVAL_SEED_BASE, and every worker seed range --
# used once, up front, to draw the fixed context sample (see load_molhiv_context_and_query).

OUTPUT_ROOT = PAPER_DIR / "dev" / "output" / "limix_backbone_gnn_pooling_multi_dataset_multi_view_pooler_multi_task_flat_lr_fit_test"


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
        default=0.0001,
        help="Flat, constant learning rate -- no warmup, no decay, no scheduler (default: 0.0001, "
        "the value validated on the single-dataset binclass sanity check to avoid the cross-entropy "
        "saturation collapse seen at higher/annealed rates -- see module docstring). Also used to size "
        "run_output_dir, so a different --lr always starts a fresh run rather than resuming one "
        "trained at a different rate.",
    )
    return parser.parse_args()


def build_prior_config(min_molecules: int, max_molecules: int) -> dict:
    """A copy of BASE_PRIOR_CONFIG with three axes widened:
    - n_graphs (molecules/dataset): fixed single value -> log_uniform_int(min_molecules, max_molecules).
    - conv_type: fixed "sage-mean" -> choice(CONV_TYPES), matching pretrain.toml's own 5-option set.
    - task: ABSENT in BASE_PRIOR_CONFIG entirely (sample_graph_level_labels_via_virtual_node/
      apply_task were never wired together before this script) -> choice(TASK_TYPES) plus the
      quantile/n_classes/multiclass_type/p_ordered/p_reverse settings apply_task needs, matching
      pretrain.toml's own [base_config.prior.prior.values.task] block. permute_labels (also absent
      from BASE_PRIOR_CONFIG) is added alongside it since apply_task requires it. See
      _sample_prior_dataset_with_conv_type_and_task, which is what actually calls apply_task with
      this config -- the graph-level prior pathway doesn't do so on its own.
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

    prior_values["task"] = {
        "_type_": {"_distribution_": "choice", "_shared_": True, "values": TASK_TYPES},
        "quantile": {"_distribution_": "uniform", "min": 0.01, "max": 0.99},
        "multiclass_type": "rank",
        "n_classes": {"_distribution_": "uniform_int", "min": 2, "max": MAX_N_CLASSES},
        "p_ordered": 0.2,
        "p_reverse": 0.5,
    }
    prior_values["postprocessing"]["permute_labels"] = True

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
    with a lightweight multi-head dot-product attention aggregation,
    concatenated and projected back to d_out. Unchanged from
    limix_backbone_gnn_pooling_multi_dataset_multi_agg_pooler_fit_test.py --
    only the FINAL readout (MultiViewPool below) changes vs. that script; see
    the multi_view_pooler_fit_test.py module docstring for why the
    message-passing step itself is left combining branches while the readout
    doesn't.
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


class MultiViewPool(nn.Module):
    """Atom->molecule readout that, instead of collapsing mean/min/max (and
    attention) pooling into ONE combined embedding via a learned linear
    projection, emits each aggregator's pooled result as its OWN separate
    group -- letting the frozen backbone's real cross-group self-attention
    decide which view matters for a given dataset via in-context inference
    over the revealed context labels. Unchanged from
    limix_backbone_gnn_pooling_multi_dataset_multi_view_pooler_fit_test.py;
    see that file's module docstring for the full rationale.

    Emits N_POOL_VIEWS separate embed_dim-sized views per input group:
    mean, min, max (order-statistic reductions, matching gcn/sage-mean/
    sage-min/sage-max) and a learned single-head attention pooling.
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
        """A learned per-atom score, softmax-normalized WITHIN each molecule
        (a segment softmax), then used to weight-sum that molecule's atoms.
        """
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
    """Message passing (real bonds) + atom->molecule pooling, applied
    independently per feature-group with SHARED weights across groups.
    Unchanged from
    limix_backbone_gnn_pooling_multi_dataset_multi_view_pooler_fit_test.py;
    see that file's module docstring for the full rationale.
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
            h = atom_embeddings_grouped[:, g, :].float()  # graph ops want float32
            for conv, norm in zip(self.convs, self.norms):
                h = self.dropout(F.relu(norm(conv(graph, h))))
            pooled_views_per_group.append(self.pool(h, molecule_id, n_molecules))  # (n_molecules, N_POOL_VIEWS, embed_dim)

        # (n_molecules, n_groups, N_POOL_VIEWS, embed_dim) -> flatten (group, view)
        # into ONE expanded "groups" axis the frozen backbone attends over.
        pooled = torch.stack(pooled_views_per_group, dim=1)
        pooled = pooled.reshape(n_molecules, n_groups * N_POOL_VIEWS, embed_dim)
        return pooled.to(dtype)


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def compute_loss(pred: torch.Tensor, target: torch.Tensor, task_type: TaskType) -> torch.Tensor:
    """MSE for regression (pred/target: (n_query,) floats); cross-entropy for both binclass and
    multiclass (pred: (n_query, MAX_N_CLASSES) raw, UNSLICED logits; target: (n_query,) long class
    ids) -- matches bin/graphpfn/pretrain.py:593-604 (both classification task types share the
    same cross_entropy call; binclass is just 2-class cross-entropy), EXCEPT label_smoothing=
    LABEL_SMOOTHING is added here -- see module docstring for why (caps the loss's target
    confidence so cross-entropy's gradient can't fully saturate, the fix validated on the
    single-dataset binclass sanity check). No slicing needed here (unlike _task_metric below):
    target class ids are always < n_classes <= MAX_N_CLASSES, so the unused wider output slots
    for datasets with fewer classes simply never receive gradient signal.
    """
    if task_type == TaskType.REGRESSION:
        return F.mse_loss(pred, target)
    return F.cross_entropy(pred, target, label_smoothing=LABEL_SMOOTHING)


def _task_metric(pred: torch.Tensor, target: torch.Tensor, task_type: TaskType, n_classes: int | None) -> tuple[str, float]:
    """Primary held-out score per task_type, mirroring the real pretrain pipeline's per-task-type
    Score choice (lib/graph/data.py's get_score: R2 for regression, accuracy for multiclass,
    average precision for binclass) -- all oriented "higher is better", so per-dataset scores stay
    roughly comparable across a mixed-task-type eval batch even though the underlying metric
    differs. `pred` is the model's raw (unsliced, MAX_N_CLASSES-wide) cls_output for classification
    tasks -- sliced here to the dataset's OWN n_classes before softmax/argmax, exactly like
    bin/graphpfn/pretrain.py's evaluate_dataset does, so untrained/irrelevant extra output slots
    never contaminate the metric.
    """
    if task_type == TaskType.REGRESSION:
        return "r2", r2_score(pred, target)
    if task_type == TaskType.MULTICLASS:
        assert n_classes is not None
        probs = pred[:, :n_classes].softmax(dim=-1)
        accuracy = (probs.argmax(dim=-1) == target).float().mean().item()
        return "accuracy", accuracy
    # binclass
    probs = pred[:, :2].softmax(dim=-1)[:, 1]
    ap = sklearn.metrics.average_precision_score(target.cpu().numpy(), probs.detach().cpu().float().numpy())
    return "ap", float(ap)


def save_checkpoint(
    path: Path,
    step: int,
    pooler_without_ddp: nn.Module,
    pooler_ema: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_held_out_score: float,
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
            "best_held_out_score": best_held_out_score,
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
    y_norm: torch.Tensor  # (n_molecules,) -- standardized regression target, OR {0,...,n_classes-1}
    # class ids for binclass/multiclass (never re-standardized, see _sample_raw_dataset)
    eval_pos_molecules: int  # first `eval_pos_molecules` indices (post-reorder) are context/train
    conv_type: str  # which of CONV_TYPES generated this dataset's label -- for the per-conv_type
    # held-out score breakdown (evaluate_held_out), so a low average can be told apart from
    # "uniformly mediocre" vs. "some conv_types (e.g. gt) still poorly fit while others are strong."
    task_type: TaskType  # regression/binclass/multiclass -- selects y_type (cls vs reg branch) in
    # forward_pass, and which loss/metric applies (see compute_loss/_task_metric).
    n_classes: int | None  # only set (2..MAX_N_CLASSES) for TaskType.MULTICLASS -- binclass is
    # always exactly 2 classes, regression has none; used to slice the model's fixed
    # MAX_N_CLASSES-wide cls_output down to the classes actually in use for the held-out metric.


def _check_label_sanity(y_norm: torch.Tensor, task_type: TaskType, n_classes: int | None, n_train: int) -> None:
    """Post-reorder classification sanity checks, mirroring the classification-only branch of
    lib/graphpfn/prior/checks.py's check_dataset (features/train-ratio sanity is already handled
    elsewhere in this molecule-count regime, so only the class-related checks are ported here):
    every sampled class must appear in BOTH the context (first n_train molecules, post-reorder) and
    query splits, and the realized label set must have exactly the expected number of distinct
    classes. Regression never raises. Called from _sample_raw_dataset's retry loop below --
    task-type sampling (unlike a regression-only script) can occasionally draw a binclass/
    multiclass dataset where a rare class fails to land in both splits, especially at
    --min-molecules' lower end combined with a large sampled n_classes.
    """
    if task_type == TaskType.BINCLASS:
        check_n_classes(y_norm, 2)
        check_class_coverage(y_norm, n_train)
    elif task_type == TaskType.MULTICLASS:
        assert n_classes is not None
        check_n_classes(y_norm, n_classes)
        check_class_coverage(y_norm, n_train)


def _sample_prior_dataset_with_conv_type_and_task(
    prior_config: dict,
) -> tuple[torch.Tensor, dgl.DGLGraph, torch.Tensor, str, TaskType, int | None]:
    """Same body as lib/graphpfn/prior/graph_level.py's sample_graph_level_dataset, but also
    resolves conv_type AND applies task-type conversion. The public function doesn't expose
    conv_type, and (per sample_graph_level_labels_via_virtual_node's own docstring) the graph-level
    pathway only ever produces a raw continuous label -- unlike the node-level pathway
    (lib/graphpfn/prior/priors/graph_then_attributes.py:49-52), it never calls apply_task itself.
    Calling sample_configs a second time to recover conv_type/task config would consume RNG state
    again and desync from the dataset actually generated, so the whole body is duplicated here
    instead of wrapping the original -- and apply_task
    (lib/graphpfn/prior/postprocessing/task.py) is invoked directly, exactly like the node-level
    pathway does, to widen this script beyond regression-only.
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

    task_config = config["task"]
    task_type = TaskType(task_config["_type_"])
    y_per_molecule = apply_task(y_per_molecule, task_config, config["postprocessing"]["permute_labels"])
    n_classes = task_config["n_classes"] if task_type == TaskType.MULTICLASS else None

    return atom_features, graph, y_per_molecule, config["scm"]["conv_type"], task_type, n_classes


def _sample_raw_dataset(features_per_group: int, prior_config: dict) -> dict:
    """The CPU-bound, model-free half of dataset prep: sample one synthetic
    molecule dataset from the prior, reorder it context-first, and pad
    features to a features_per_group multiple. No GPU/model access at all --
    this is exactly what's safe to run inside a DataLoader worker
    subprocess, in parallel, while the main process is busy on the GPU.

    Wrapped in a retry loop (N_LABEL_SANITY_RETRIES): a binclass/multiclass
    draw whose realized classes don't fully cover both the context and query
    splits (_check_label_sanity) is discarded and resampled from scratch,
    exactly like lib/graphpfn/prior/sampler.py's own
    _sample_dataset_with_retry does for the real pipeline.

    Returns a plain dict of CPU tensors, NOT a SampledDataset -- the graph is
    passed as raw (src, dst) edge tensors rather than a dgl.DGLGraph, since
    reconstructing it with dgl.graph(...) in the main process (see
    encode_raw_dataset_on_gpu) is simpler and cheaper than relying on
    DGLGraph objects surviving multiprocessing IPC/pickling.
    """
    for attempt in range(N_LABEL_SANITY_RETRIES):
        atom_features, graph, y_per_molecule, conv_type, task_type, n_classes = (
            _sample_prior_dataset_with_conv_type_and_task(prior_config)
        )

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
        if task_type == TaskType.REGRESSION:
            y_mean = y_reordered[:n_train].mean()
            y_std = y_reordered[:n_train].std().clamp(min=1e-6)
            y_norm = (y_reordered - y_mean) / y_std
        else:
            # apply_task already produced the final {0,1}/{0,...,n_classes-1} class ids -- no
            # further standardization (that would corrupt class identity).
            y_norm = y_reordered

        try:
            _check_label_sanity(y_norm, task_type, n_classes, n_train)
            break
        except SanityCheckError:
            if attempt == N_LABEL_SANITY_RETRIES - 1:
                raise

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
        "task_type": task_type,
        "n_classes": n_classes,
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
        task_type=raw["task_type"],
        n_classes=raw["n_classes"],
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
    exact same molecules/labels/task_type, and its seed range
    (EVAL_SEED_BASE+) never overlaps any unseeded training draw, so it never
    leaks into training.
    """
    with _temporary_rng_seed(EVAL_SEED_BASE + idx):
        return sample_and_prepare_dataset(model, device, prior_config)


def _graph_has_isolated_atom(data) -> bool:
    """True if any atom in `data` has zero bonds (in-degree 0 in edge_index). The synthetic prior's
    own molecule sampler always yields fully-connected graphs, so PoolingGNN's mean/min/max
    aggregation (dgl.ops.copy_u_mean/min/max in MultiAggregatorConv) was never built to handle a
    node with zero neighbors -- it reduces to NaN/inf there, which then poisons every OTHER
    molecule's prediction too once the frozen backbone's cross-molecule attention mixes it in. A
    real dataset like ogbg-molhiv does contain a handful of these (~5.5% of graphs, mostly salts
    with a disconnected counter-ion) -- filtered out at load time (load_molhiv_context_and_query)
    rather than changing the (already-duplicated-here, not-yet-trained-on) model architecture.
    """
    n_atoms = data.x.shape[0]
    in_degree = torch.zeros(n_atoms, dtype=torch.long)
    in_degree.scatter_add_(0, data.edge_index[1], torch.ones(data.edge_index.shape[1], dtype=torch.long))
    return bool((in_degree == 0).any())


def load_molhiv_context_and_query(n_context: int, seed: int) -> tuple[list, list]:
    """Loads the real ogbg-molhiv dataset once and returns (context_graphs, query_graphs): a FIXED,
    seeded random n_context-sized sample of the train split (the in-context "demonstrations") and
    the FULL (filtered) test split as query -- a real, out-of-synthetic-prior molecular benchmark
    tracked alongside the synthetic held-out eval as training progresses. Both splits first drop
    any graph with an isolated atom (see _graph_has_isolated_atom) -- ~5.5% of ogbg-molhiv, a
    pattern the synthetic prior never produces and PoolingGNN's aggregation isn't built to handle;
    "full test split" therefore means every REMAINING test graph, not literally every original one.
    Uses PygGraphPropPredDataset directly (no torch_geometric DataLoader --
    _molhiv_graphs_to_raw_dataset below builds its own custom batched (atom_features, edges,
    molecule_id) tensors, the same way _sample_raw_dataset does for synthetic data, so the
    DataLoader's batching machinery isn't needed).
    """
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=str(MOLHIV_ROOT))
    split_idx = dataset.get_idx_split()

    train_idx = np.array([i for i in split_idx["train"].numpy() if not _graph_has_isolated_atom(dataset[int(i)])])
    test_idx = np.array([i for i in split_idx["test"].numpy() if not _graph_has_isolated_atom(dataset[int(i)])])

    rng = np.random.RandomState(seed)
    context_idx = rng.choice(train_idx, size=n_context, replace=False)
    context_graphs = [dataset[int(i)] for i in context_idx]
    query_graphs = [dataset[int(i)] for i in test_idx]
    return context_graphs, query_graphs


def _molhiv_graphs_to_raw_dataset(context_graphs: list, query_graphs: list, features_per_group: int) -> dict:
    """Builds the same raw-dict shape _sample_raw_dataset produces for synthetic data (atom
    features/edges/molecule_id/y_norm/eval_pos_*), but from real ogbg-molhiv PyG graphs instead of
    the synthetic prior -- context graphs first (revealed labels), query graphs after (masked),
    exactly like the context-first reorder _sample_raw_dataset does for synthetic molecules. Node
    features are OGB's raw 9-dim integer atom encoding (atomic number, chirality, degree, ...) --
    passed through as plain floats, NOT run through process_features (that pipeline's categorical/
    permutation logic is specific to the synthetic prior's own feature semantics); the frozen
    backbone's own x_preprocess/encoder_x already handles feature normalization internally, exactly
    like it does for the synthetic path's features_per_group-padded features. edge_index is used
    as-is (OGB's mol conversion already emits both directions per bond, matching how this script's
    own synthetic molecule graphs are built).
    """
    graphs = context_graphs + query_graphs
    n_context = len(context_graphs)

    atom_features_list = []
    edges_src_list = []
    edges_dst_list = []
    molecule_id_list = []
    y_list = []
    atom_offset = 0
    for mol_idx, data in enumerate(graphs):
        n_atoms_mol = data.x.shape[0]
        atom_features_list.append(data.x.float())
        edges_src_list.append(data.edge_index[0] + atom_offset)
        edges_dst_list.append(data.edge_index[1] + atom_offset)
        molecule_id_list.append(torch.full((n_atoms_mol,), mol_idx, dtype=torch.long))
        y_list.append(data.y.view(-1).float())
        atom_offset += n_atoms_mol

    atom_features = torch.cat(atom_features_list, dim=0)
    edges_src = torch.cat(edges_src_list, dim=0)
    edges_dst = torch.cat(edges_dst_list, dim=0)
    molecule_id = torch.cat(molecule_id_list, dim=0)
    y_norm = torch.cat(y_list, dim=0)  # already {0.0, 1.0} class ids -- no standardization, matching
    # how _sample_raw_dataset also leaves apply_task's classification output untouched.

    n_atoms, n_features = atom_features.shape
    feature_to_add = n_features % features_per_group
    if feature_to_add > 0:
        pad = torch.zeros(n_atoms, features_per_group - feature_to_add, dtype=atom_features.dtype)
        atom_features = torch.cat([atom_features, pad], dim=-1)

    eval_pos_atoms = int(sum(g.x.shape[0] for g in context_graphs))

    return {
        "atom_features": atom_features,
        "edges_src": edges_src,
        "edges_dst": edges_dst,
        "n_atoms": n_atoms,
        "molecule_id": molecule_id,
        "n_molecules": len(graphs),
        "y_norm": y_norm,
        "eval_pos_atoms": eval_pos_atoms,
        "eval_pos_molecules": n_context,
        "conv_type": "molhiv",
        "task_type": TaskType.BINCLASS,
        "n_classes": None,
    }


def evaluate_molhiv(
    model: nn.Module,
    device: torch.device,
    pooler_ema: nn.Module,
    context_graphs: list,
    query_graphs: list,
) -> float:
    """Runs one ICL forward pass with `context_graphs` as the revealed in-context examples and ALL
    of `query_graphs` (the full ogbg-molhiv test split) as the masked query, through the SAME
    encode_raw_dataset_on_gpu/forward_pass pipeline used for the synthetic held-out eval -- just
    with a real molecular dataset instead of a fresh synthetic draw, and a FIXED context (sampled
    once in main(), not per-call) so successive calls during training track improvement on the
    exact same slice. Returns the ROC-AUC over the query predictions -- OGB's own metric for
    ogbg-molhiv (unlike synthetic binclass draws, which use average precision -- see _task_metric --
    ROC-AUC is used here to stay directly comparable to published ogbg-molhiv results).
    """
    pooler_ema.eval()
    with torch.no_grad():
        raw = _molhiv_graphs_to_raw_dataset(context_graphs, query_graphs, model.features_per_group)
        dataset = encode_raw_dataset_on_gpu(model, device, raw)
        pred, target = forward_pass(model, dataset, pooler_ema)
        probs = pred[:, :2].softmax(dim=-1)[:, 1]
        auc = roc_auc_score(target.cpu().numpy(), probs.detach().cpu().float().numpy())
    return float(auc)


def forward_pass(model: nn.Module, dataset: SampledDataset, pooler_module: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Runs the frozen backbone's add_embeddings -> mixed_y_embedding ->
    transformer_encoder -> y_decoder chain over ALL of `dataset`'s molecules
    (context labels revealed, query masked to NaN). y_type routes every
    position through the backbone's cls or reg branch as ONE fixed choice
    for the whole dataset (vendor/limix/model/transformer.py's
    mixed_y_embedding/y_decoder: y_type==0 -> cls, ==1 -> reg), matching
    dataset.task_type. Returns (query_predictions, query_targets):
    - regression: predictions are the model's raw reg_output, (n_query,)
      float scalars; targets are the standardized continuous label.
    - binclass/multiclass: predictions are the model's raw, UNSLICED
      MAX_N_CLASSES-wide cls_output logits, (n_query, MAX_N_CLASSES);
      targets are long class ids. compute_loss/_task_metric each slice/
      interpret these differently (loss needs no slicing; the eval metric
      slices to n_classes -- see _task_metric).
    """
    eval_pos = dataset.eval_pos_molecules
    pooled_grouped = pooler_module(dataset.graph, dataset.atom_embeddings_grouped, dataset.molecule_id, dataset.n_molecules)
    pooled_x = pooled_grouped.unsqueeze(0).to(next(model.encoder_x.parameters()).dtype)

    embedded_x = model.add_embeddings(pooled_x)  # (1, n_molecules, n_groups * N_POOL_VIEWS, embed_dim)

    y_local = dataset.y_norm.unsqueeze(0).unsqueeze(-1).clone()  # (1, n_molecules, 1)
    y_dict = {"data": y_local}
    y_dict["data"][:, eval_pos:] = torch.nan
    is_classification = dataset.task_type != TaskType.REGRESSION
    y_type = torch.zeros_like(y_dict["data"]) if is_classification else torch.ones_like(y_dict["data"])
    embedded_y = model.mixed_y_embedding(y_dict, y_type=y_type, eval_pos=eval_pos)

    embedded_all = torch.cat((embedded_x, embedded_y.unsqueeze(2).to(embedded_x.dtype)), dim=2)
    encoder_out = model.transformer_encoder(embedded_all, feature_atten_mask=None, eval_pos=eval_pos)[0]
    encoder_out = model.encoder_out_norm(encoder_out)

    test_encoder_out = encoder_out[:, eval_pos:, -1]
    test_y_type = y_type[:, eval_pos:]
    cls_output, reg_output = model.y_decoder(test_encoder_out, test_y_type)

    target = dataset.y_norm[eval_pos:]
    if is_classification:
        pred = cls_output.float().squeeze(0)  # (n_query, MAX_N_CLASSES)
        target = target.long()
    else:
        pred = reg_output.float().squeeze(0).squeeze(-1)  # (n_query,)
    return pred, target


def main() -> None:
    # tqdm.write() lines otherwise sit in a stdout buffer and can lag far
    # behind tqdm's own progress-bar redraws (a different, more eager flush
    # path) when stdout isn't a TTY (e.g. redirected to a SLURM .out file) --
    # see module docstring.
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
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
    assert MAX_N_CLASSES <= model.decoder_config["num_classes"], (
        f"MAX_N_CLASSES={MAX_N_CLASSES} exceeds the checkpoint's own fixed cls_output width "
        f"({model.decoder_config['num_classes']}) -- lower MAX_N_CLASSES to match."
    )
    if is_main:
        print(f"features_per_group={model.features_per_group}, embed_dim={embed_dim}, nlayers={model.nlayers}")
        print(
            f"PoolingGNN: MultiAggregatorConv (mean/min/max/attention, {N_ATTN_HEADS} heads) x "
            f"{N_GNN_LAYERS} layers + MultiViewPool readout ({N_POOL_VIEWS} separate groups/input-group "
            f"instead of a combined single embedding)"
        )
        print(
            f"Training on a FRESH synthetic dataset every micro-step "
            f"({args.min_molecules}-{args.max_molecules} molecules, log-uniform per draw; "
            f"conv_type in {CONV_TYPES}, graph_conv_ratio=1.0 fixed; task in {TASK_TYPES} "
            f"(n_classes 2-{MAX_N_CLASSES} for multiclass); causal.enabled=False fixed) -- "
            f"lr={args.lr}, {N_GRADIENT_ACCUMULATION_STEPS} per optimizer step, {args.n_steps} steps "
            f"total = {args.n_steps * N_GRADIENT_ACCUMULATION_STEPS} datasets), evaluating on "
            f"{N_EVAL_DATASETS} fixed-seed held-out datasets every {EVAL_EVERY} steps..."
        )
        print(
            f"Prefetching training datasets with {args.n_sampler_workers} background workers/rank "
            f"(prefetch_factor={args.prefetch_factor})..."
        )
        print(f"Run output dir: {run_output_dir}")

    # Real-data ICL eval (ogbg-molhiv) is main-process-only, exactly like evaluate_held_out --
    # loaded once here (not per eval call) so the context sample stays fixed across the whole run.
    molhiv_context_graphs, molhiv_query_graphs = None, None
    if is_main:
        print(
            f"Loading real ogbg-molhiv dataset from {MOLHIV_ROOT} for ICL eval "
            f"({MOLHIV_N_CONTEXT} random train context, full test query, both splits filtered to "
            f"drop isolated-atom graphs -- see _graph_has_isolated_atom)..."
        )
        molhiv_context_graphs, molhiv_query_graphs = load_molhiv_context_and_query(MOLHIV_N_CONTEXT, MOLHIV_EVAL_SEED)
        print(
            f"  {len(molhiv_context_graphs)} context graphs, {len(molhiv_query_graphs)} query graphs "
            f"({len(molhiv_context_graphs) + len(molhiv_query_graphs)} molecules total per eval call)"
        )

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
    ema_multi_avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(decay=EMA_DECAY)
    pooler_ema = torch.optim.swa_utils.AveragedModel(pooler_without_ddp, device, multi_avg_fn=ema_multi_avg_fn)

    # Resume support: every rank independently reads the same checkpoint file
    # (safe -- it's read-only from here, and all ranks need identical model/
    # optimizer/scheduler state). Only valid for continuing an INTERRUPTED
    # run of this SAME config; see build_run_paths' docstring for why this
    # can't extend n_steps (that always gets a fresh output dir instead).
    start_step = 1
    best_held_out_score = -float("inf")
    best_step = -1
    if pooler_checkpoint_path.exists():
        checkpoint = load_checkpoint(pooler_checkpoint_path, device)
        pooler_without_ddp.load_state_dict(checkpoint["pooler"])
        pooler_ema.load_state_dict(checkpoint["pooler_ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_held_out_score = checkpoint["best_held_out_score"]
        best_step = checkpoint["best_step"]
        start_step = checkpoint["step"] + 1
        if is_main:
            print(
                f"Resumed from {pooler_checkpoint_path} at step {checkpoint['step']} "
                f"(best held-out score so far: {best_held_out_score:.4f} @ step {best_step})"
            )

    def evaluate_held_out() -> float:
        """Mean held-out score over N_EVAL_DATASETS fixed-seed draws, using the EMA weights --
        genuinely unseen data (a disjoint seed range from every training draw). Each dataset's own
        score is R2 (regression), accuracy (multiclass), or average precision (binclass) --
        see _task_metric. These live on different natural scales, so this mean is a rough
        "did held-out performance improve" signal, not a calibrated single number -- the printed
        per-dataset breakdown (conv_type, task_type, n_molecules, metric name, value) is what
        actually tells apart "uniformly mediocre" from "some conv_types/task_types still poorly fit
        while others are already strong." Only ever called from within an `if is_main:` block.
        """
        pooler_ema.eval()
        per_dataset = []
        with torch.no_grad():
            for idx in range(N_EVAL_DATASETS):
                dataset = sample_eval_dataset(model, device, idx, prior_config)
                pred, target = forward_pass(model, dataset, pooler_ema)
                metric_name, score = _task_metric(pred, target, dataset.task_type, dataset.n_classes)
                per_dataset.append((dataset.conv_type, dataset.task_type, dataset.n_molecules, metric_name, score))
        breakdown = " | ".join(
            f"{conv_type}/{task_type.value}(n={n_molecules})={metric_name}:{score:.3f}"
            for conv_type, task_type, n_molecules, metric_name, score in per_dataset
        )
        tqdm.write(f"  held-out per-dataset: {breakdown}")
        return float(np.mean([score for *_, score in per_dataset]))

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
                loss = compute_loss(pred, target, dataset.task_type)
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

        held_out_score = None
        molhiv_auc = None
        if step % EVAL_EVERY == 0 or step == args.n_steps:
            if is_main:
                held_out_score = evaluate_held_out()
                if held_out_score > best_held_out_score:
                    best_held_out_score = held_out_score
                    best_step = step
                tqdm.write(
                    f"step {step:5d} | held-out score (EMA, {N_EVAL_DATASETS} datasets, mixed metrics) = "
                    f"{held_out_score:.4f}"
                )
                molhiv_auc = evaluate_molhiv(model, device, pooler_ema, molhiv_context_graphs, molhiv_query_graphs)
                tqdm.write(
                    f"step {step:5d} | molhiv ICL ROC-AUC (EMA, {len(molhiv_query_graphs)} query graphs, "
                    f"{MOLHIV_N_CONTEXT} context) = {molhiv_auc:.4f}"
                )
                save_checkpoint(
                    pooler_checkpoint_path,
                    step,
                    pooler_without_ddp,
                    pooler_ema,
                    optimizer,
                    best_held_out_score,
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
                    "held_out_score": held_out_score,
                    "molhiv_auc": molhiv_auc,
                },
            )

    if is_main:
        print(f"\nbest held-out score = {best_held_out_score:.4f} at step {best_step}")
        print(f"checkpoint: {pooler_checkpoint_path}")
        print(f"per-step log: {training_log_path}")


if __name__ == "__main__":
    main()
