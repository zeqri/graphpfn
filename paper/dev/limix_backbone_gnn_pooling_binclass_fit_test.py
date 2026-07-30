"""Binary-classification counterpart to the very first script in this
family (limix_backbone_gnn_pooling_fit_test.py, since removed/reorganized
from dev/ -- this rebuilds its design from earlier in this session): sample
ONE fixed synthetic dataset, train repeatedly on it for many epochs, and
check whether the model can fit it at all. That single-dataset design was
never a generalization test (it's closer to full-batch gradient descent
converging on one stationary target) -- it's a minimal sanity check for
"does the mechanism work at all," which is exactly what's needed right now:
the multi-dataset multi-task run
(limix_backbone_gnn_pooling_multi_dataset_multi_view_pooler_multi_task_fit_test.py)
showed loss/held-out metrics not clearly improving even after the
vanishing-gradient issue was fixed (lower LR), and it's not yet known
whether that's "needs more time/tuning" or "something about how
classification is wired up doesn't work at all," given every earlier
single/multi-dataset sanity check in this session was regression-only.

Two axes that add real difficulty in the multi-task multi-dataset script are
FIXED here, deliberately, to isolate the classification mechanism itself:
- conv_type: left at BASE_PRIOR_CONFIG's own default ("sage-mean") instead
  of widened to all 5 options -- removes the aggregation-mismatch question
  entirely for this test.
- quantile (lib/graphpfn/prior/postprocessing/task.py's regression_to_binclass:
  the fraction of molecules that land in class 1) is FIXED at 0.5 (balanced),
  not sampled from pretrain.toml's uniform(0.01, 0.99) -- removes the
  class-imbalance-driven "predict the majority class" shortcut that's a
  likely contributor to the multi-task script's saturating-gradient
  collapse. p_reverse is fixed to 0.0 and permute_labels to False for the
  same reason: fewer moving parts, nothing here should affect whether the
  model CAN fit at all (permutation/reversal are label-invariant
  relabelings).

Architecture (PoolingGNN/MultiAggregatorConv/MultiViewPool) is copied
verbatim from
limix_backbone_gnn_pooling_multi_dataset_multi_view_pooler_multi_task_fit_test.py
-- deliberately the SAME architecture as the failing run, not the original
family's simple SAGEConv-mean pooler, since the question is whether THIS
architecture can fit classification at all.

No DDP/prefetching/checkpointing here, unlike the multi-dataset scripts --
a single fixed dataset trained for N_EPOCHS is cheap enough (one dataset
sampled once, one frozen encode, then pure GPU compute per epoch) that none
of that machinery is needed; this is a single-process script, closer in
spirit to the standalone diagnose_conv_type_size_r2*.py scripts than to the
multi-dataset training scripts.

Usage:
    python dev/limix_backbone_gnn_pooling_binclass_fit_test.py
(Single-process, single-GPU-or-CPU.)
"""

from __future__ import annotations

import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent.parent
if str(PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_DIR))

import copy  # noqa: E402
import dgl  # noqa: E402
import dgl.nn.pytorch as dglnn  # noqa: E402
import numpy as np  # noqa: E402
import sklearn.metrics  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tqdm import tqdm  # noqa: E402

import lib  # noqa: E402
import lib.deep  # noqa: E402
from lib.graphpfn.prior.graph_level import sample_graph_level_dataset  # noqa: E402
from lib.graphpfn.prior.postprocessing import apply_task  # noqa: E402
from lib.graphpfn.prior.checks import SanityCheckError, check_class_coverage, check_n_classes  # noqa: E402
from lib.util import TaskType  # noqa: E402
from vendor.limix.utils.loading import load_model  # noqa: E402
from dev.limix_encoder_pooling_probe import BASE_PRIOR_CONFIG  # noqa: E402

SEED = 0
CHECKPOINT_PATH = PAPER_DIR / "checkpoints" / "LimiX-16M.ckpt"
TRAIN_FRACTION = 0.8

# Fixed, not sampled -- see module docstring for why both are pinned for
# this specific sanity check.
FIXED_QUANTILE = 0.5  # balanced 50/50 class split (regression_to_binclass's own
# semantics: `quantile` IS the resulting class-1 fraction before any reversal)
FIXED_P_REVERSE = 0.0
N_SAMPLE_RETRIES = 10  # resample-from-scratch budget if the single fixed draw fails
# class-coverage sanity (context and query must each contain both classes) -- should
# rarely matter at TRAIN_FRACTION=0.8 with a balanced 50/50 split, but molecule count
# is itself random (BASE_PRIOR_CONFIG's own graph.n_nodes distribution).

# BASE_PRIOR_CONFIG's own default is 3000 molecules/dataset -- combined with
# MultiViewPool's 4x-expanded group count (n_groups -> n_groups * N_POOL_VIEWS), the
# frozen backbone's cross-group feature-attention cost (roughly quadratic in group
# count, see limix_backbone_gnn_pooling_multi_dataset_multi_view_pooler_fit_test.py's
# module docstring) is far more memory-hungry here than it ever was for the original
# family's single-view pooler at the same molecule count -- OOM'd at the default size.
# Shrinking to 300 (same reduction used earlier in this session for the multi-dataset
# scripts) trades a shorter in-context sequence for actually fitting in memory.
N_MOLECULES = 2000

N_GNN_LAYERS = 3
N_ATTN_HEADS = 4  # embed_dim=192 (LimiX-16M) / 4 = 48 per head -- message-passing attention branch only
N_POOL_VIEWS = 4  # mean, min, max, learned-attention -- see MultiViewPool
DROPOUT = 0.0  # matches the real graph-adapter modules' own default (layers.py:
# GraphPFNGraphAttentionModule/GraphPFNMLPModule both default dropout=0.0)

N_EPOCHS = 1000
LOG_EVERY = 20

# Optimizer/clipping/EMA settings taken directly from the real pretraining config
# (exp/graphpfn/pretrain/main/pretrain.toml), same as every earlier script in this
# family -- EXCEPT the LR schedule (see FLAT_LR below).
OPTIMIZER_TYPE = "AdamW"
WEIGHT_DECAY = 0.1
GRADIENT_CLIPPING_NORM = 1.0
EMA_DECAY = 0.98

# Deliberately NO scheduler/warmup at all here (unlike every other script in this
# family, which cosine-ramps up to a peak) -- observed directly in this script's own
# runs: collapse (grad_norm dying, loss freezing near chance level) happened once LR got
# reasonably close to ITS peak regardless of what that peak was (0.001 collapsed by
# epoch ~100, 0.0005 only delayed it to ~epoch 280-300, 0.0005+label_smoothing still
# showed the same weak-gradient pattern by ~epoch 300). The common factor across all
# three is RAMPING UP toward a peak, not the peak's specific value -- so this tests
# whether never ramping there at all (a low, flat, constant LR from step 1) avoids
# triggering it, rather than just delaying it again. No lr_scheduler object at all;
# optimizer's LR is set once and never changed.
FLAT_LR = 0.0001

# Cross-entropy's gradient vanishes as predicted probabilities approach 0/1 (right or
# wrong) -- observed directly in this script's own runs: loss/grad_norm froze completely
# partway through training regardless of peak LR (0.001 collapsed by epoch ~100, 0.0005
# only delayed it to ~epoch 280-300), meaning "find a safe peak LR" wasn't a real fix,
# just a later trigger of the same saturation. LABEL_SMOOTHING caps how confident the
# target is allowed to be (e.g. 0.9 instead of 1.0 for the true class), which keeps the
# gradient from ever fully saturating -- a direct fix for the actual mechanism, not
# another indirect LR-schedule workaround.
LABEL_SMOOTHING = 0.1


def build_prior_config() -> dict:
    """BASE_PRIOR_CONFIG (conv_type stays at its own default "sage-mean") with
    n_graphs (molecule count) shrunk to N_MOLECULES -- see that constant's
    comment for why -- plus a "task" section forcing binclass with a FIXED
    quantile/p_reverse, mirroring
    limix_backbone_gnn_pooling_multi_dataset_multi_view_pooler_multi_task_fit_test.py's
    build_prior_config but with every distribution collapsed to a single
    value -- see module docstring for why.
    """
    prior_config = copy.deepcopy(BASE_PRIOR_CONFIG)
    prior_values = prior_config["prior"]["values"][0]

    n_graphs_dist = {"_distribution_": "choice", "values": [N_MOLECULES]}
    prior_values["graph"]["sampler"]["values"][0]["n_graphs"] = n_graphs_dist
    prior_values["graph"]["n_nodes"] = n_graphs_dist  # cosmetic only -- the multi-graph
    # sampler (lib/graphpfn/prior/graphs/multi_graph.py) actually sizes datasets from
    # n_graphs/base_n_nodes above, not this top-level field, but keeping it in sync
    # avoids a misleading stale value.

    prior_values["task"] = {
        "_type_": {"_distribution_": "choice", "_shared_": True, "values": ["binclass"]},
        "quantile": {"_distribution_": "choice", "values": [FIXED_QUANTILE]},
        "multiclass_type": "rank",
        "n_classes": {"_distribution_": "choice", "values": [2]},  # unused for binclass, kept for apply_task's schema
        "p_ordered": 1.0,
        "p_reverse": {"_distribution_": "choice", "values": [FIXED_P_REVERSE]},
    }
    prior_values["postprocessing"]["permute_labels"] = False
    return prior_config


class MultiAggregatorConv(nn.Module):
    """Per-layer message passing combining mean/min/max neighbor reductions
    with a lightweight multi-head dot-product attention aggregation,
    concatenated and projected back to d_out. Copied verbatim from
    limix_backbone_gnn_pooling_multi_dataset_multi_view_pooler_multi_task_fit_test.py.
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
    limix_backbone_gnn_pooling_multi_dataset_multi_view_pooler_multi_task_fit_test.py.
    """

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
        scores = self.attn_score(atom_out).squeeze(-1)

        scores_max = torch.full((n_molecules,), float("-inf"), device=device, dtype=dtype).scatter_reduce(
            0, molecule_id, scores, reduce="amax", include_self=False
        )
        shifted = (scores - scores_max[molecule_id]).exp()
        denom = torch.zeros(n_molecules, device=device, dtype=dtype).index_add(0, molecule_id, shifted).clamp(min=1e-12)
        weights = shifted / denom[molecule_id]

        weighted = atom_out * weights.unsqueeze(-1)
        return torch.zeros(n_molecules, atom_out.shape[-1], device=device, dtype=dtype).index_add(0, molecule_id, weighted)


class PoolingGNN(nn.Module):
    """Identical architecture to
    limix_backbone_gnn_pooling_multi_dataset_multi_view_pooler_multi_task_fit_test.py's
    PoolingGNN -- deliberately the SAME architecture as the failing
    multi-dataset run, see module docstring.
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

        pooled = torch.stack(pooled_views_per_group, dim=1)
        pooled = pooled.reshape(n_molecules, n_groups * N_POOL_VIEWS, embed_dim)
        return pooled.to(dtype)


def _sample_fixed_binclass_dataset(prior_config: dict) -> tuple[torch.Tensor, dgl.DGLGraph, torch.Tensor]:
    """Samples ONE dataset from the prior and applies apply_task to get a
    binary label, retrying from scratch (N_SAMPLE_RETRIES) if the realized
    split fails class-coverage sanity -- same pattern as the multi-task
    multi-dataset script's _check_label_sanity, just without the
    per-micro-step retry loop's complexity since this only ever runs once.
    """
    for attempt in range(N_SAMPLE_RETRIES):
        atom_features, graph, y_per_molecule, _ = sample_graph_level_dataset(prior_config)
        # sample_graph_level_dataset's own train_ratio return value is discarded (matches
        # every earlier script in this family) -- context/query split uses our own
        # TRAIN_FRACTION below instead.

        task_config = prior_config["prior"]["values"][0]["task"]
        # apply_task needs the actual resolved config (quantile/p_reverse are fixed to a
        # single value here, so no resampling is needed to "resolve" them -- unlike
        # conv_type in the multi-dataset scripts, nothing here varies per draw).
        resolved_task_config = {
            "_type_": "binclass",
            "quantile": FIXED_QUANTILE,
            "p_reverse": FIXED_P_REVERSE,
        }
        y_per_molecule = apply_task(y_per_molecule, resolved_task_config, permute_labels=False)

        n_molecules = y_per_molecule.shape[0]
        n_train = int(n_molecules * TRAIN_FRACTION)
        # check_class_coverage/check_n_classes expect the FIRST n_train entries to be the
        # train/context split -- sample_graph_level_dataset's own molecule order hasn't
        # been shuffled yet at this point, so do that first (mirrors _sample_raw_dataset's
        # context-first reorder in every other script, just simplified to run once).
        perm = np.random.permutation(n_molecules)
        mol_order = np.concatenate([perm[:n_train], perm[n_train:]])
        y_reordered = y_per_molecule[torch.from_numpy(mol_order)]

        try:
            check_n_classes(y_reordered, 2)
            check_class_coverage(y_reordered, n_train)
            return atom_features, graph, y_per_molecule
        except SanityCheckError:
            if attempt == N_SAMPLE_RETRIES - 1:
                raise
    raise AssertionError("unreachable")


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """pred: (n_query, 2) raw cls_output logits (already sliced to the 2 binclass
    columns); target: (n_query,) long {0,1} class ids. Reports accuracy AND average
    precision -- accuracy is the more intuitive "is it fitting" signal for a BALANCED
    single dataset (unlike the multi-task script's mixed-imbalance eval set, where AP is
    the more meaningful metric); both are cheap to compute so both are logged.
    """
    probs = pred.softmax(dim=-1)[:, 1]
    preds_binary = (probs > 0.5).long()
    accuracy = (preds_binary == target).float().mean().item()
    ap = sklearn.metrics.average_precision_score(target.cpu().numpy(), probs.detach().cpu().float().numpy())
    return {"accuracy": accuracy, "ap": float(ap)}


def main() -> None:
    # tqdm.write()'s eval-line prints were sitting in a stdout buffer and never reaching
    # the SLURM .out file until the buffer filled/the process exited -- the progress-bar
    # redraws themselves flushed fine (a separate code path), which made it look like no
    # eval had run at all even well past several LOG_EVERY checkpoints. Line-buffering
    # stdout here fixes this regardless of how the script is launched (sbatch, srun,
    # interactively), unlike relying on `python -u` at the call site.
    sys.stdout.reconfigure(line_buffering=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = lib.get_device()
    print(f"Device: {device}")
    print(f"Loading real LimiX-16M checkpoint from {CHECKPOINT_PATH}...")
    model = load_model(str(CHECKPOINT_PATH), mask_prediction=False)
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    features_per_group = model.features_per_group
    embed_dim = model.embed_dim
    print(f"features_per_group={features_per_group}, embed_dim={embed_dim}, nlayers={model.nlayers}")

    print(
        f"Sampling ONE fixed dataset (conv_type=BASE_PRIOR_CONFIG default 'sage-mean', "
        f"task=binclass, quantile={FIXED_QUANTILE} FIXED, p_reverse={FIXED_P_REVERSE} FIXED)..."
    )
    prior_config = build_prior_config()
    atom_features, graph, y_per_molecule = _sample_fixed_binclass_dataset(prior_config)
    n_features = atom_features.shape[-1]

    # Context-first molecule reorder (unbatch/rebatch keeps each molecule's own atoms +
    # real bonds + features together, without manual edge-index math) -- same pattern as
    # every earlier script in this family.
    graph.ndata["feat"] = atom_features
    mol_graphs = dgl.unbatch(graph)
    n_molecules = len(mol_graphs)
    perm = np.random.permutation(n_molecules)
    n_train = int(n_molecules * TRAIN_FRACTION)
    mol_order = np.concatenate([perm[:n_train], perm[n_train:]])

    reordered_graph = dgl.batch([mol_graphs[i] for i in mol_order]).to(device)
    atom_features_reordered = reordered_graph.ndata["feat"]
    counts_reordered = reordered_graph.batch_num_nodes()
    molecule_id = torch.repeat_interleave(torch.arange(n_molecules, device=device), counts_reordered)
    eval_pos_atoms = int(counts_reordered[:n_train].sum().item())
    eval_pos_molecules = n_train

    y_reordered = y_per_molecule[torch.from_numpy(mol_order)].to(device).long()
    n_pos_train = int((y_reordered[:n_train] == 1).sum().item())
    n_pos_query = int((y_reordered[n_train:] == 1).sum().item())
    print(
        f"n_atoms={graph.num_nodes()}, n_molecules={n_molecules}, n_features={n_features}, "
        f"train: {n_train} molecules ({n_pos_train} positive, {n_train - n_pos_train} negative), "
        f"query: {n_molecules - n_train} molecules ({n_pos_query} positive, "
        f"{n_molecules - n_train - n_pos_query} negative)"
    )

    n_atoms = atom_features_reordered.shape[0]
    feature_to_add = n_features % features_per_group
    if feature_to_add > 0:
        pad = torch.zeros(
            n_atoms, features_per_group - feature_to_add, dtype=atom_features_reordered.dtype, device=device
        )
        atom_features_reordered = torch.cat([atom_features_reordered, pad], dim=-1)
    n_groups = atom_features_reordered.shape[-1] // features_per_group

    def encode_atoms_frozen() -> torch.Tensor:
        """x_preprocess -> process_4_x -> encoder_x, all frozen, no_grad -- features don't
        change across epochs, so this runs exactly once (matches the original
        single-dataset script's own design).
        """
        x = atom_features_reordered.unsqueeze(0)
        x_dict = {"data": x, "mask": torch.isnan(x).to(torch.int32)}
        x_dict = {k: v.reshape(1, n_atoms, n_groups, features_per_group) for k, v in x_dict.items()}
        x_dict["eval_pos"] = eval_pos_atoms
        with torch.no_grad():
            preprocessed = model.x_preprocess(x_dict)
            preprocessed = model.process_4_x(preprocessed)
            x_encoder_result = model.encoder_x(preprocessed)
        return x_encoder_result["data"].squeeze(0)

    print("Encoding atoms with frozen encoder_x (one-time, features don't change across epochs)...")
    atom_embeddings_grouped = encode_atoms_frozen()

    pooler = PoolingGNN(embed_dim=embed_dim, n_layers=N_GNN_LAYERS, dropout=DROPOUT, n_heads=N_ATTN_HEADS).to(device)

    params = lib.deep.make_parameter_groups(pooler)
    optimizer = lib.deep.make_optimizer(type=OPTIMIZER_TYPE, lr=FLAT_LR, weight_decay=WEIGHT_DECAY, params=params)
    # No lr_scheduler at all -- see FLAT_LR's comment for why.
    ema_multi_avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(decay=EMA_DECAY)
    pooler_ema = torch.optim.swa_utils.AveragedModel(pooler, device, multi_avg_fn=ema_multi_avg_fn)

    def forward_pass(pooler_module: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs the frozen backbone's add_embeddings -> mixed_y_embedding ->
        transformer_encoder -> y_decoder chain over ALL molecules (context labels
        revealed, query masked to NaN), routed through the CLASSIFICATION branch
        (y_type=0) the whole time, unlike every earlier regression-only script in this
        family (y_type=1). Returns (query_cls_logits, query_targets).
        """
        pooled_grouped = pooler_module(reordered_graph, atom_embeddings_grouped, molecule_id, n_molecules)
        pooled_x = pooled_grouped.unsqueeze(0).to(next(model.encoder_x.parameters()).dtype)

        embedded_x = model.add_embeddings(pooled_x)

        y_local = y_reordered.float().unsqueeze(0).unsqueeze(-1).clone()
        y_dict = {"data": y_local}
        y_dict["data"][:, eval_pos_molecules:] = torch.nan
        y_type = torch.zeros_like(y_dict["data"])  # 0 == classification branch
        embedded_y = model.mixed_y_embedding(y_dict, y_type=y_type, eval_pos=eval_pos_molecules)

        embedded_all = torch.cat((embedded_x, embedded_y.unsqueeze(2).to(embedded_x.dtype)), dim=2)
        encoder_out = model.transformer_encoder(embedded_all, feature_atten_mask=None, eval_pos=eval_pos_molecules)[0]
        encoder_out = model.encoder_out_norm(encoder_out)

        test_encoder_out = encoder_out[:, eval_pos_molecules:, -1]
        test_y_type = y_type[:, eval_pos_molecules:]
        cls_output, _ = model.y_decoder(test_encoder_out, test_y_type)
        pred = cls_output.float().squeeze(0)[:, :2]  # (n_query, 2) -- binclass always exactly 2 classes
        target = y_reordered[eval_pos_molecules:]
        return pred, target

    best_test_acc = -float("inf")
    best_epoch = -1

    iterator = tqdm(range(N_EPOCHS), desc="training")
    for epoch in iterator:
        pooler.train()
        optimizer.zero_grad()
        pred, target = forward_pass(pooler)
        loss = F.cross_entropy(pred, target, label_smoothing=LABEL_SMOOTHING)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(pooler.parameters(), GRADIENT_CLIPPING_NORM)
        optimizer.step()
        pooler_ema.update_parameters(pooler)
        # No lr_scheduler.step() -- LR stays flat at FLAT_LR the whole run, see its comment.

        iterator.set_postfix(loss=loss.item(), grad_norm=grad_norm.item(), lr=lib.deep.get_lr(optimizer))

        if epoch % LOG_EVERY == 0 or epoch == N_EPOCHS - 1:
            pooler_ema.eval()
            with torch.no_grad():
                pred, target = forward_pass(pooler_ema)
                metrics = compute_metrics(pred, target)
            if metrics["accuracy"] > best_test_acc:
                best_test_acc = metrics["accuracy"]
                best_epoch = epoch
            tqdm.write(
                f"epoch {epoch:4d} | loss {loss.item():.4f} | grad_norm {grad_norm.item():.4f} | "
                f"lr {lib.deep.get_lr(optimizer):.6f} | test accuracy (EMA) {metrics['accuracy']:.4f} | "
                f"test AP (EMA) {metrics['ap']:.4f}"
            )

    print(f"\nbest test accuracy = {best_test_acc:.4f} at epoch {best_epoch} (early-stopping proxy)")


if __name__ == "__main__":
    main()
