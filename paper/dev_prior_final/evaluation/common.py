"""Shared building blocks for every evaluation script in this directory (BACE / BBBP / ClinTox /
SIDER classification; ESOL / FreeSolv / Lipophilicity / ZINC / AQSOL regression).

Self-contained: the PoolingGNN architecture and its forward pass live here (identical to the one
trained by ../train_pooler.py, so its checkpoints load strict), together with
checkpoint loading, dataset + pretrained-embedding loading, the atom-feature -> raw-dict builder,
the embedding feature-group splice, LimiX-style permutation ensembling and metrics.

NO ABSOLUTE PATHS. The pooler checkpoint comes from the command line; every other default is
resolved relative to this file:
  * dataset roots       -> <repo>/datasets/{moleculenet,zinc,aqsol}   (--moleculenet/zinc/aqsol-root)
  * backbone checkpoint -> <paper>/checkpoints/LimiX-16M.ckpt
  * embeddings root     -> <repo>/embeddings/<Molbert|MolDeBERTa>/<dataset>/
  * LimiX configs       -> <paper>/vendor/limix/config/

EMBEDDINGS LAYOUT (same for Molbert and MolDeBERTa):
  <embeddings_root>/<model>/<dataset>/<dataset>_embeddings_{train,valid,test}.npy
  <embeddings_root>/<model>/<dataset>/<dataset>_meta.json
where meta["split_idx"][split] lists SOURCE-CSV ROW indices, row r of the split's .npy is CSV row
split_idx[split][r], and anything that failed to parse/encode is already excluded from both.
PyG's MoleculeNet drops CSV rows that yield zero atoms, so CSV row != PyG index in general;
`csv_row_to_pyg_index` rebuilds that mapping by replaying PyG's own CSV parse, so it works for every
dataset without any dataset-specific failure bookkeeping.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PAPER_DIR = EVAL_DIR.parent.parent  # evaluation/ -> dev_prior_final/ -> paper/
REPO_DIR = PAPER_DIR.parent
for _p in (str(PAPER_DIR), str(EVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dgl  # noqa: E402
import numpy as np  # noqa: E402
import sklearn.metrics  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch_geometric.datasets import MoleculeNet  # noqa: E402

from lib.graphpfn.model import (  # noqa: E402
    GraphPFNGraphAttentionModule,
    GraphPFNMLPModule,
    GraphPFNResidualModule,
)
from vendor.limix.utils.loading import load_model  # noqa: E402

from atom_features import (  # noqa: E402
    N_EXTRA_FEATURES,
    N_X_PLUS_EXTRA_FEATURES,
    extra_atom_features_from_smiles,
    x_plus_extra_atom_features,
)

DEFAULT_BACKBONE_CHECKPOINT = PAPER_DIR / "checkpoints" / "LimiX-16M.ckpt"
DEFAULT_EMBEDDINGS_ROOT = REPO_DIR / "embeddings"
DEFAULT_DATASETS_ROOT = REPO_DIR / "datasets"  # holds moleculenet/, zinc/, aqsol/
DEFAULT_OUTPUT_ROOT = EVAL_DIR / "outputs"
CLS_INFERENCE_CONFIG = PAPER_DIR / "vendor" / "limix" / "config" / "cls_default_16M_retrieval.json"
REG_INFERENCE_CONFIG = PAPER_DIR / "vendor" / "limix" / "config" / "reg_default_16M_retrieval.json"

EMBEDDING_MODELS = ["Molbert", "MolDeBERTa"]
SPLITS = ("train", "valid", "test")

# Architecture constants -- must match train_pooler.py.
N_GNN_LAYERS = 1
N_ATTN_HEADS = 4
N_POOL_VIEWS = 4
DROPOUT = 0.0
GRAPH_ADAPTER_N_HEADS = 4
EMA_DECAY = 0.98  # only shapes the AveragedModel wrapper; eval never calls update_parameters().

# Fixed divisor on the classification logits.
TEMPERATURE = 4

# Permutation-ensemble members per pooler pipeline (same count as the TabPFN/TabICL baselines' --n-estimators).
DEFAULT_LIMIX_N_MEMBERS = 8

# Atom-feature variants: name -> (per-molecule featurizer, number of columns).
FEATURE_VARIANTS = {
    "extra_features_only": (
        lambda data: extra_atom_features_from_smiles(data.smiles, n_atoms_expected=data.x.shape[0]),
        N_EXTRA_FEATURES,
    ),
    "x_plus_extra_features": (x_plus_extra_atom_features, N_X_PLUS_EXTRA_FEATURES),
}


def autocast_ctx(device: torch.device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------

DATASET_ROOT_HELP = {
    "moleculenet": "torch_geometric MoleculeNet root (the directory holding <dataset>/raw, <dataset>/processed).",
    "zinc": "torch_geometric ZINC root (holds raw/{atom,bond}_dict.pickle and subset/).",
    "aqsol": "torch_geometric AQSOL root (holds raw/, processed/ and data_curated.csv).",
}


def add_common_args(parser: argparse.ArgumentParser, dataset_root: str = "moleculenet") -> None:
    """Arguments shared by every evaluation script. `dataset_root` names the
    --<dataset_root>-root flag (moleculenet / zinc / aqsol), defaulting to <repo>/datasets/<name>."""
    parser.add_argument(f"--{dataset_root}-root", type=Path, default=DEFAULT_DATASETS_ROOT / dataset_root,
                        help=DATASET_ROOT_HELP[dataset_root] + " (default: %(default)s)")
    parser.add_argument("--pooler-checkpoint", type=Path, required=True, help="Pooler checkpoint (.pt) to evaluate.")
    parser.add_argument("--backbone-checkpoint", type=Path, default=DEFAULT_BACKBONE_CHECKPOINT,
                        help="Frozen LimiX backbone checkpoint (default: %(default)s).")
    parser.add_argument("--embedding-model", choices=EMBEDDING_MODELS, required=True,
                        help="Which pretrained molecule embeddings to splice in / feed to raw LimiX.")
    parser.add_argument("--embeddings-root", type=Path, default=DEFAULT_EMBEDDINGS_ROOT,
                        help="Root holding <embedding-model>/<dataset>/ (default: %(default)s).")
    parser.add_argument("--no-ema", action="store_true", help="Use the raw (non-EMA) pooler weights instead of the EMA weights.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-embeddings", "--skip-molebert", dest="skip_embeddings", action="store_true",
                        help="Skip BOTH embedding-augmented pooler pipelines AND the embedding+raw-LimiX pipeline.")
    parser.add_argument("--skip-embedding-limix", "--skip-molebert-limix", dest="skip_embedding_limix", action="store_true",
                        help="Skip only the embedding + raw-LimiX tabular-ICL pipeline (GPU-only).")
    parser.add_argument("--limix-seed", type=int, default=0,
                        help="Seed for LimiXPredictor (raw-LimiX pipeline) and for the pooler pipelines' "
                        "feature/class-permutation ensemble members.")
    parser.add_argument("--limix-n-members", type=int, default=DEFAULT_LIMIX_N_MEMBERS,
                        help="Permutation ensemble members averaged per pooler pipeline (default: %(default)s).")
    parser.add_argument("--output-json", type=Path, default=None,
                        help=f"Metrics JSON (default: {DEFAULT_OUTPUT_ROOT}/<embedding-model>/<name>.json).")


def resolve_output_json(args: argparse.Namespace, name: str) -> Path:
    path = args.output_json or (DEFAULT_OUTPUT_ROOT / args.embedding_model / f"{name}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------------------------------
# PoolingGNN architecture (identical to train_pooler.py)
# --------------------------------------------------------------------------------------------------

class MultiAggregatorConv(nn.Module):
    """Per-layer message passing combining mean/min/max neighbor reductions with a lightweight
    multi-head dot-product attention aggregation."""

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
        # attn_score is autocast-touched while atom_out (a DGL op output) is not -- force a dtype
        # match before scatter_reduce, which requires one.
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
    """One transformer block's graph-attention + MLP adapter, built from lib.graphpfn.model."""

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
    """Message passing (real bonds) + atom->molecule pooling + per-block RealGraphAdapter refinement.
    Only the parameters live here -- the forward is `pooling_gnn_forward` below, which additionally
    supports splicing in a pretrained-embedding feature group."""

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


def pooling_gnn_forward(
    pooler: PoolingGNN, model: nn.Module, graph: dgl.DGLGraph, augmented_graph: dgl.DGLGraph,
    atom_embeddings_grouped: torch.Tensor, molecule_id: torch.Tensor, n_molecules: int,
    embedded_y: torch.Tensor, eval_pos: int, embeddings_grouped: torch.Tensor | None = None,
) -> torch.Tensor:
    """train_pooler.py's PoolingGNN.forward with ONE insertion: if embeddings_grouped
    (n_molecules, n_emb_groups, embed_dim) is given, it is concatenated onto `pooled` right after
    output_proj. embeddings_grouped=None reproduces the checkpoint's own forward exactly."""
    n_atoms, n_groups, embed_dim = atom_embeddings_grouped.shape
    assert len(pooler.graph_adapters) == len(model.transformer_encoder.layers), (
        f"{len(pooler.graph_adapters)=} != {len(model.transformer_encoder.layers)=}"
    )

    pooled_views_per_group = []
    atom_hidden_per_group = []
    for g in range(n_groups):
        h = atom_embeddings_grouped[:, g, :].float()
        for conv, norm in zip(pooler.convs, pooler.norms):
            h = pooler.dropout(F.relu(norm(conv(graph, h))))
        atom_hidden_per_group.append(h)
        pooled_views_per_group.append(pooler.pool(h, molecule_id, n_molecules))

    pooled = torch.stack(pooled_views_per_group, dim=1)
    pooled = pooled.reshape(n_molecules, n_groups * N_POOL_VIEWS, embed_dim)
    pooled = pooler.output_proj(pooled)

    if embeddings_grouped is not None:
        pooled = torch.cat([pooled, embeddings_grouped.to(pooled.dtype)], dim=1)

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
        combined_hidden = pooler.graph_adapters[i](augmented_graph, combined_hidden)

        current_vnode = combined_hidden[n_atoms:]
        new_vnode_slot = current_vnode.to(x.dtype).unsqueeze(0).unsqueeze(2)
        x = torch.cat([before, new_vnode_slot, after], dim=2)

    return x


# --------------------------------------------------------------------------------------------------
# model / checkpoint loading
# --------------------------------------------------------------------------------------------------

def load_backbone(backbone_checkpoint: Path, device: torch.device) -> nn.Module:
    print(f"Loading frozen LimiX backbone from {backbone_checkpoint}")
    model = load_model(str(backbone_checkpoint), mask_prediction=False).to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def load_pooler(pooler_checkpoint: Path, model: nn.Module, device: torch.device, use_ema: bool) -> tuple[PoolingGNN, dict]:
    """Rebuilds the PoolingGNN the checkpoint was trained with and loads its (EMA or raw) weights.
    Returns (pooler, checkpoint_info)."""
    print(f"Loading pooler checkpoint from {pooler_checkpoint} (use_ema={use_ema})")
    checkpoint = torch.load(pooler_checkpoint, map_location=device)
    pooler = PoolingGNN(
        embed_dim=model.embed_dim, n_layers=N_GNN_LAYERS, dropout=DROPOUT,
        n_transformer_blocks=model.nlayers, n_heads=N_ATTN_HEADS,
    ).to(device)

    if use_ema:
        ema_multi_avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(decay=EMA_DECAY)
        pooler_ema = torch.optim.swa_utils.AveragedModel(pooler, device, multi_avg_fn=ema_multi_avg_fn)
        pooler_ema.load_state_dict(checkpoint["pooler_ema"])
        pooler = pooler_ema.module
    else:
        pooler.load_state_dict(checkpoint["pooler"], strict=True)
    pooler.eval()

    best_key = next((k for k in ("best_held_out_ap", "best_held_out_roc_auc", "best_held_out_mae") if k in checkpoint), None)
    info = {
        "step": checkpoint.get("step"),
        "best_step": checkpoint.get("best_step"),
        "best_metric": best_key,
        "best_value": checkpoint.get(best_key) if best_key else None,
        "use_ema": use_ema,
    }
    best_str = f"{best_key}={info['best_value']:.4f} @ step {info['best_step']}" if best_key else "(no best_* field)"
    print(f"  checkpoint step {info['step']} -- {best_str}")
    return pooler, info


# --------------------------------------------------------------------------------------------------
# datasets + pretrained embeddings
# --------------------------------------------------------------------------------------------------

def load_moleculenet(moleculenet_root: Path, name: str) -> MoleculeNet:
    print(f"Loading MoleculeNet '{name}' from {moleculenet_root / name}")
    return MoleculeNet(root=str(moleculenet_root), name=name)


def csv_row_to_pyg_index(ds: MoleculeNet, name: str) -> dict[int, int]:
    """Maps every source-CSV row index (the index space of meta["split_idx"]) to its position in
    the PyG dataset. Replays MoleculeNet.process's own line parsing on ds.raw_paths[0], then walks
    rows and PyG entries together: PyG only ever DROPS rows (zero-atom molecules), never reorders,
    so a row is kept iff its SMILES equals the next unmatched PyG entry's SMILES."""
    smiles_col = MoleculeNet.names[name.lower()][3]
    with open(ds.raw_paths[0]) as f:
        lines = [x for x in f.read().split("\n")[1:-1] if len(x) > 0]
    row_smiles = [re.sub(r"\".*\"", "", line).split(",")[smiles_col] for line in lines]

    mapping: dict[int, int] = {}
    j = 0
    for row, smi in enumerate(row_smiles):
        if j < len(ds) and ds[j].smiles == smi:
            mapping[row] = j
            j += 1
    if j != len(ds):
        raise RuntimeError(
            f"Could only align {j}/{len(ds)} PyG '{name}' molecules to the {len(row_smiles)} rows of "
            f"{ds.raw_paths[0]} -- the processed dataset does not match its raw CSV."
        )
    return mapping


def embeddings_dir_for(args: argparse.Namespace, dataset: str) -> Path:
    return args.embeddings_root / args.embedding_model / dataset


def load_meta(embeddings_dir: Path, dataset: str) -> dict:
    with open(embeddings_dir / f"{dataset}_meta.json") as f:
        return json.load(f)


def load_split(
    ds: MoleculeNet, meta: dict, csv_to_pyg: dict[int, int], embeddings_dir: Path, dataset: str, split: str,
) -> tuple[list, np.ndarray]:
    """(list[Data], (n, emb_dim) float32 embedding rows) for one split, aligned row-for-row via
    meta["split_idx"][split] (CSV rows)."""
    csv_rows = meta["split_idx"][split]
    emb = np.load(embeddings_dir / f"{dataset}_embeddings_{split}.npy").astype(np.float32)
    if emb.shape[0] != len(csv_rows):
        raise RuntimeError(f"{dataset}/{split}: {emb.shape[0]} embedding rows != {len(csv_rows)} split_idx entries")

    missing = [c for c in csv_rows if c not in csv_to_pyg]
    if missing:
        raise RuntimeError(
            f"{dataset}/{split}: {len(missing)} split_idx CSV row(s) were dropped by PyG MoleculeNet "
            f"(zero atoms), e.g. {missing[:5]} -- the meta should already exclude these."
        )
    examples = [ds[csv_to_pyg[c]] for c in csv_rows]
    return examples, emb


def load_all_splits(args: argparse.Namespace, dataset: str) -> tuple[MoleculeNet, dict, dict, dict, Path]:
    """Loads the PyG dataset and train/valid/test (examples, embeddings) for `dataset`. Returns
    (ds, meta, examples_by_split, emb_by_split, embeddings_dir)."""
    ds = load_moleculenet(args.moleculenet_root, dataset)
    emb_dir = embeddings_dir_for(args, dataset)
    meta = load_meta(emb_dir, dataset)
    csv_to_pyg = csv_row_to_pyg_index(ds, dataset)
    examples, emb = {}, {}
    for split in SPLITS:
        examples[split], emb[split] = load_split(ds, meta, csv_to_pyg, emb_dir, dataset, split)
    print(f"  {args.embedding_model} embeddings from {emb_dir}: "
          + ", ".join(f"{s}={emb[s].shape}" for s in SPLITS)
          + f" (PyG holds {len(ds)} of {meta['shape'][0]} CSV rows)")
    return ds, meta, examples, emb, emb_dir


def load_local_index_split(pyg_split_ds, meta: dict, embeddings_dir: Path, dataset: str, split: str) -> tuple[list, np.ndarray, list[int]]:
    """For datasets pre-split at the source (ZINC, AQSOL): meta lists LOCAL indices into each split's
    own PyG dataset -- under "split_idx" (or "graph_index" in the MolDeBERTa AQSOL meta) -- and row r
    of <dataset>_embeddings_<split>.npy is pyg_split_ds[indices[r]]. Returns (examples, embeddings,
    indices)."""
    key = "split_idx" if "split_idx" in meta else "graph_index"
    indices = [int(i) for i in meta[key][split]]
    emb = np.load(embeddings_dir / f"{dataset}_embeddings_{split}.npy").astype(np.float32)
    if emb.shape[0] != len(indices):
        raise RuntimeError(f"{dataset}/{split}: {emb.shape[0]} embedding rows != {len(indices)} meta['{key}'] entries")
    return [pyg_split_ds[i] for i in indices], emb, indices


def labels_matrix(examples: list) -> np.ndarray:
    """(n, n_tasks) float64 labels straight from each Data's own `y` (NaN = missing)."""
    return np.stack([d.y.reshape(-1).cpu().numpy() for d in examples]).astype(np.float64)


# --------------------------------------------------------------------------------------------------
# atom features -> raw dict -> encoded dataset
# --------------------------------------------------------------------------------------------------

def featurize(examples: list, variant: str) -> list[torch.Tensor]:
    """Per-molecule atom-feature matrices for one FEATURE_VARIANTS entry -- computed once and reused
    across every permutation member / ensemble run."""
    feature_fn, _ = FEATURE_VARIANTS[variant]
    return [feature_fn(d) for d in examples]


def isolated_atom_local_indices(edge_index: torch.Tensor, n_atoms: int) -> torch.Tensor:
    """Local indices of zero-in-degree atoms (they get a self-loop before message passing)."""
    in_degree = torch.zeros(n_atoms, dtype=torch.long)
    if edge_index.numel() > 0:
        in_degree.scatter_add_(0, edge_index[1], torch.ones(edge_index.shape[1], dtype=torch.long))
    return (in_degree == 0).nonzero(as_tuple=False).flatten()


def examples_to_raw(
    examples: list, atom_features: list[torch.Tensor], y: torch.Tensor, n_context: int,
    features_per_group: int, perm: np.ndarray | None = None,
) -> dict:
    """Topology + precomputed atom features + per-molecule labels -> raw dict. `examples` (and the
    aligned `atom_features` / `y`) must be in final context-first order. Zero-degree atoms get a
    self-loop. `perm`, if given, permutes the real feature columns before features_per_group
    padding (mirrors LimiX's FeatureShuffler)."""
    edges_src_list: list[torch.Tensor] = []
    edges_dst_list: list[torch.Tensor] = []
    molecule_id_list: list[torch.Tensor] = []

    atom_offset = 0
    eval_pos_atoms = None
    for mol_idx, (data, x) in enumerate(zip(examples, atom_features)):
        if mol_idx == n_context:
            eval_pos_atoms = atom_offset
        n_atoms_mol = x.shape[0]
        if n_atoms_mol == 0:
            raise ValueError(f"molecule {mol_idx} ({getattr(data, 'smiles', '?')!r}) has zero atoms")

        edge_index = getattr(data, "edge_index", None)
        if edge_index is None or edge_index.numel() == 0:
            edge_index = torch.empty(2, 0, dtype=torch.long)
        edge_index = edge_index.long()
        edges_src_list.append(edge_index[0] + atom_offset)
        edges_dst_list.append(edge_index[1] + atom_offset)

        isolated_local = isolated_atom_local_indices(edge_index, n_atoms_mol)
        if isolated_local.numel() > 0:
            edges_src_list.append(isolated_local + atom_offset)
            edges_dst_list.append(isolated_local + atom_offset)

        molecule_id_list.append(torch.full((n_atoms_mol,), mol_idx, dtype=torch.long))
        atom_offset += n_atoms_mol
    if eval_pos_atoms is None:
        eval_pos_atoms = atom_offset

    x_all = torch.cat(atom_features, dim=0).float()
    if perm is not None:
        x_all = x_all[:, perm]

    n_atoms, n_features = x_all.shape
    feature_to_add = n_features % features_per_group
    if feature_to_add > 0:
        pad = torch.zeros(n_atoms, features_per_group - feature_to_add, dtype=x_all.dtype)
        x_all = torch.cat([x_all, pad], dim=-1)

    return {
        "atom_features": x_all,
        "edges_src": torch.cat(edges_src_list, dim=0),
        "edges_dst": torch.cat(edges_dst_list, dim=0),
        "n_atoms": n_atoms,
        "molecule_id": torch.cat(molecule_id_list, dim=0),
        "n_molecules": len(examples),
        "y": y.float(),
        "eval_pos_atoms": eval_pos_atoms,
        "eval_pos_molecules": n_context,
    }


@dataclasses.dataclass
class EncodedDataset:
    graph: dgl.DGLGraph
    augmented_graph: dgl.DGLGraph
    atom_embeddings_grouped: torch.Tensor
    molecule_id: torch.Tensor
    n_molecules: int
    y: torch.Tensor  # class ids (classification) or standardized targets (regression).
    eval_pos_molecules: int


def encode_raw_dataset_on_gpu(model: nn.Module, device: torch.device, raw: dict) -> EncodedDataset:
    """Builds the bond graph + atom<->virtual-node augmented graph and runs the frozen LimiX feature
    encoder over the atom features."""
    n_atoms = raw["n_atoms"]
    n_molecules = raw["n_molecules"]
    graph = dgl.graph((raw["edges_src"], raw["edges_dst"]), num_nodes=n_atoms).to(device)
    atom_features = raw["atom_features"].to(device)
    molecule_id = raw["molecule_id"].to(device)

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
    x_dict["eval_pos"] = raw["eval_pos_atoms"]
    with torch.no_grad():
        preprocessed = model.x_preprocess(x_dict)
        preprocessed = model.process_4_x(preprocessed)
        x_encoder_result = model.encoder_x(preprocessed)

    return EncodedDataset(
        graph=graph, augmented_graph=augmented_graph,
        atom_embeddings_grouped=x_encoder_result["data"].squeeze(0),
        molecule_id=molecule_id, n_molecules=n_molecules, y=raw["y"].to(device),
        eval_pos_molecules=raw["eval_pos_molecules"],
    )


def encode_embeddings_as_groups(
    model: nn.Module, embeddings: torch.Tensor, eval_pos_molecules: int,
) -> torch.Tensor:
    """Runs the frozen LimiX feature encoder over pretrained molecule embeddings, one row per
    MOLECULE. Returns (n_molecules, n_embedding_groups, embed_dim)."""
    features_per_group = model.features_per_group
    n_molecules, n_features = embeddings.shape
    pad = (-n_features) % features_per_group
    if pad:
        embeddings = F.pad(embeddings, (0, pad))
    n_groups = embeddings.shape[-1] // features_per_group

    x = embeddings.unsqueeze(0)
    x_dict = {"data": x, "mask": torch.isnan(x).to(torch.int32)}
    x_dict = {k: v.reshape(1, n_molecules, n_groups, features_per_group) for k, v in x_dict.items()}
    x_dict["eval_pos"] = eval_pos_molecules
    with torch.no_grad():
        preprocessed = model.x_preprocess(x_dict)
        preprocessed = model.process_4_x(preprocessed)
        encoder_result = model.encoder_x(preprocessed)
    return encoder_result["data"].squeeze(0)


# --------------------------------------------------------------------------------------------------
# forward passes
# --------------------------------------------------------------------------------------------------

def _forward_encoder(model, dataset: EncodedDataset, pooler, embeddings_grouped, y_type_value: int):
    eval_pos = dataset.eval_pos_molecules
    y_local = dataset.y.unsqueeze(0).unsqueeze(-1).clone()
    y_dict = {"data": y_local}
    y_dict["data"][:, eval_pos:] = torch.nan
    y_type = torch.full_like(y_dict["data"], float(y_type_value))
    embedded_y = model.mixed_y_embedding(y_dict, y_type=y_type, eval_pos=eval_pos)

    x = pooling_gnn_forward(
        pooler, model, dataset.graph, dataset.augmented_graph, dataset.atom_embeddings_grouped,
        dataset.molecule_id, dataset.n_molecules, embedded_y, eval_pos, embeddings_grouped,
    )
    encoder_out = model.encoder_out_norm(x)
    return model.y_decoder(encoder_out[:, eval_pos:, -1], y_type[:, eval_pos:])


def forward_pass_cls(
    model: nn.Module, dataset: EncodedDataset, pooler: PoolingGNN, embeddings_grouped: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classification branch (y_type=0). Returns ((n_query, 2) logits / TEMPERATURE, (n_query,) target)."""
    cls_output, _ = _forward_encoder(model, dataset, pooler, embeddings_grouped, y_type_value=0)
    pred = cls_output.float().squeeze(0)[:, :2] / TEMPERATURE
    return pred, dataset.y[dataset.eval_pos_molecules:].long()


def forward_pass_reg(
    model: nn.Module, dataset: EncodedDataset, pooler: PoolingGNN, embeddings_grouped: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Regression branch (y_type=1). Returns ((n_query,) standardized prediction, (n_query,) target)."""
    _, reg_output = _forward_encoder(model, dataset, pooler, embeddings_grouped, y_type_value=1)
    pred = reg_output.float().squeeze(0).squeeze(-1)
    return pred, dataset.y[dataset.eval_pos_molecules:]


# --------------------------------------------------------------------------------------------------
# LimiX-style permutation ensembling
# --------------------------------------------------------------------------------------------------

def seeded_feature_perm(n_features: int, seed: int, member_idx: int) -> np.ndarray:
    """Same operation as vendor/limix/inference/preprocess.py's FeatureShuffler (seeded column
    permutation, identical for every row)."""
    return np.random.default_rng(seed + member_idx).permutation(n_features)


def seeded_class_perm(seed: int, member_idx: int) -> np.ndarray:
    """Binary counterpart of LimiXPredictor._predict_cls's per-estimator class permutation; the
    large offset decorrelates it from seeded_feature_perm."""
    return np.random.default_rng(seed + member_idx + 1_000_003).permutation(2)


def build_permuted_datasets(
    model: nn.Module, device: torch.device, examples: list, atom_features: list[torch.Tensor],
    y: torch.Tensor, n_context: int, n_members: int, seed: int,
) -> list[EncodedDataset]:
    """One encoded dataset per ensemble member, each with its own seeded feature-column
    permutation. Labels are only carried along -- callers may swap them per task via y_override."""
    n_features = atom_features[0].shape[1]
    datasets = []
    for member_idx in range(n_members):
        perm = seeded_feature_perm(n_features, seed, member_idx)
        with torch.no_grad(), autocast_ctx(device):
            raw = examples_to_raw(examples, atom_features, y, n_context, model.features_per_group, perm=perm)
            dataset = encode_raw_dataset_on_gpu(model, device, raw)
        assert dataset.n_molecules == len(examples) and dataset.eval_pos_molecules == n_context
        datasets.append(dataset)
    return datasets


def run_pooler_ensemble_cls(
    model: nn.Module, pooler: PoolingGNN, device: torch.device, datasets: list[EncodedDataset], seed: int,
    embeddings_grouped: torch.Tensor | None = None, y_override: torch.Tensor | None = None,
) -> tuple[np.ndarray, torch.Tensor]:
    """Class-permutation ensembling as in LimiXPredictor._predict_cls: permute context labels
    before each member's forward pass, un-permute the output columns after, average softmax
    probabilities. Returns (positive-class probability, true query labels)."""
    probs, target_ref = [], None
    for member_idx, dataset in enumerate(datasets):
        eval_pos = dataset.eval_pos_molecules
        y_base = y_override if y_override is not None else dataset.y
        class_perm_idx = torch.as_tensor(seeded_class_perm(seed, member_idx), dtype=torch.long, device=y_base.device)
        y_permuted = y_base.clone()
        y_permuted[:eval_pos] = class_perm_idx[y_base[:eval_pos].long()].to(y_permuted.dtype)
        with torch.no_grad(), autocast_ctx(device):
            pred, _ = forward_pass_cls(model, dataclasses.replace(dataset, y=y_permuted), pooler, embeddings_grouped)
        target = y_base[eval_pos:].long()
        if target_ref is None:
            target_ref = target
        else:
            assert torch.equal(target_ref, target)
        probs.append(pred[:, class_perm_idx].softmax(dim=-1)[:, 1].detach().cpu().float().numpy())
    return np.mean(probs, axis=0), target_ref


# --------------------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------------------

def clf_metrics(pos_prob, y_true) -> dict[str, float]:
    """ROC-AUC / AP / accuracy (threshold 0.5) from the positive-class probability; NaN ROC-AUC
    when a slice is single-class."""
    pos_prob = np.asarray(pos_prob, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    pred = (pos_prob > 0.5).astype(int)
    roc_auc = float(sklearn.metrics.roc_auc_score(y_true, pos_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    ap = float(sklearn.metrics.average_precision_score(y_true, pos_prob))
    return {"roc_auc": roc_auc, "ap": ap, "accuracy": float((pred == y_true).mean())}


def three_way_metrics(pos_prob: np.ndarray, target, n_valid: int) -> dict[str, dict[str, float]]:
    """Query order is [valid..., test...]."""
    tgt = target.detach().cpu().numpy() if isinstance(target, torch.Tensor) else np.asarray(target)
    return {
        "valid": clf_metrics(pos_prob[:n_valid], tgt[:n_valid]),
        "test": clf_metrics(pos_prob[n_valid:], tgt[n_valid:]),
        "combined": clf_metrics(pos_prob, tgt),
    }


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(target)) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(target))))


REGRESSION_METRICS = {"rmse": rmse, "mae": mae}


def summarize_regression(pred_runs: np.ndarray, target: np.ndarray, metric: str = "rmse") -> dict:
    """Ensemble-averaged and per-run `metric` (rmse | mae) + R2, in real target units."""
    fn = REGRESSION_METRICS[metric]
    n = pred_runs.shape[0]
    err_runs = np.array([fn(pred_runs[i], target) for i in range(n)])
    r2_runs = np.array([sklearn.metrics.r2_score(target, pred_runs[i]) for i in range(n)])
    ensemble_pred = pred_runs.mean(axis=0)
    return {
        f"ensemble_{metric}": fn(ensemble_pred, target),
        "ensemble_r2": float(sklearn.metrics.r2_score(target, ensemble_pred)),
        f"per_run_{metric}": err_runs.tolist(),
        f"per_run_{metric}_mean": float(err_runs.mean()),
        f"per_run_{metric}_std": float(err_runs.std()),
        "per_run_r2": r2_runs.tolist(),
        "per_run_r2_mean": float(r2_runs.mean()),
        "per_run_r2_std": float(r2_runs.std()),
        "n_ensemble": n,
    }


# --------------------------------------------------------------------------------------------------
# regression: pooler pipelines (context resample x feature permutation) and raw LimiX
# --------------------------------------------------------------------------------------------------

def run_regression_ensemble(
    model: nn.Module, pooler: PoolingGNN, device: torch.device,
    train_examples: list, test_examples: list, train_feats: list[torch.Tensor], test_feats: list[torch.Tensor],
    train_y: torch.Tensor, test_y: torch.Tensor, train_emb: torch.Tensor | None, test_emb: torch.Tensor | None,
    y_mean: float, y_std: float, max_train: int, n_ensemble: int, seed: int, test_chunk_size: int,
    limix_seed: int, limix_n_members: int, metric: str = "rmse",
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """n_ensemble runs, each with a fresh context of max_train molecules drawn without replacement
    from train (np.random.default_rng(seed)), the query scored in chunks of test_chunk_size, and
    limix_n_members seeded feature-column permutations averaged per chunk. Every member yields a
    baseline prediction and -- if embeddings are given -- an embedding-augmented one over the SAME
    molecules. Returns (baseline runs [n_ensemble, n_test], augmented runs or None, target), all in
    real target units."""
    err = REGRESSION_METRICS[metric]
    rng = np.random.default_rng(seed)
    n_context = min(max_train, len(train_examples))
    n_features = train_feats[0].shape[1]
    runs_base, runs_aug, target_real = [], [], None
    for run_idx in range(n_ensemble):
        context_idx = rng.choice(len(train_examples), size=n_context, replace=False)
        chunk_base, chunk_aug, chunk_tgt = [], [], []
        for start in range(0, len(test_examples), test_chunk_size):
            stop = start + test_chunk_size
            examples = [train_examples[i] for i in context_idx] + test_examples[start:stop]
            feats = [train_feats[i] for i in context_idx] + test_feats[start:stop]
            y_norm = (torch.cat([train_y[context_idx], test_y[start:stop]]) - y_mean) / y_std

            emb_grouped = None
            if train_emb is not None:
                emb_this = torch.cat([train_emb[context_idx], test_emb[start:stop]], dim=0).to(device)
                with torch.no_grad(), autocast_ctx(device):
                    emb_grouped = encode_embeddings_as_groups(model, emb_this, eval_pos_molecules=n_context)

            members_base, members_aug, target = [], [], None
            for member_idx in range(limix_n_members):
                perm = seeded_feature_perm(n_features, limix_seed, member_idx)
                raw = examples_to_raw(examples, feats, y_norm, n_context, model.features_per_group, perm=perm)
                dataset = encode_raw_dataset_on_gpu(model, device, raw)
                with torch.no_grad(), autocast_ctx(device):
                    pred, target = forward_pass_reg(model, dataset, pooler)
                    members_base.append(pred.float().cpu().numpy())
                    if emb_grouped is not None:
                        pred_aug, _ = forward_pass_reg(model, dataset, pooler, emb_grouped)
                        members_aug.append(pred_aug.float().cpu().numpy())

            chunk_base.append(np.mean(members_base, axis=0) * y_std + y_mean)
            if members_aug:
                chunk_aug.append(np.mean(members_aug, axis=0) * y_std + y_mean)
            chunk_tgt.append(target.float().cpu().numpy() * y_std + y_mean)

        if target_real is None:
            target_real = np.concatenate(chunk_tgt)
        runs_base.append(np.concatenate(chunk_base))
        msg = f"baseline {metric.upper()}={err(runs_base[-1], target_real):.6f}"
        if chunk_aug:
            runs_aug.append(np.concatenate(chunk_aug))
            msg += f", embedding_augmented {metric.upper()}={err(runs_aug[-1], target_real):.6f}"
        print(f"  run {run_idx + 1}/{n_ensemble} done ({msg})")

    return np.stack(runs_base), (np.stack(runs_aug) if runs_aug else None), target_real


def run_embedding_limix_reg(
    device: torch.device, backbone_checkpoint: Path, X_train_full: np.ndarray, y_train_full: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray, max_train: int, n_ensemble: int, seed: int, test_chunk_size: int,
    limix_seed: int, metric: str = "rmse",
) -> dict:
    """Embeddings as plain tabular features -> raw LimiX regression retrieval ICL. `seed` drives the
    per-run context subsample (train_test_split, random_state=seed+run); `limix_seed` drives
    LimiXPredictor's own internal ensembling."""
    from sklearn.model_selection import train_test_split

    from vendor.limix.inference.predictor import LimiXPredictor

    err = REGRESSION_METRICS[metric]
    print(f"  Loading LimiXPredictor (reg retrieval) on {device}, seed={limix_seed}; "
          f"train available={len(X_train_full)}, test={len(X_test)}")
    reg = LimiXPredictor(
        device=device, model_path=str(backbone_checkpoint), inference_config=str(REG_INFERENCE_CONFIG), seed=limix_seed,
    )
    n_context = min(max_train, len(X_train_full))
    pred_runs = []
    for run_idx in range(n_ensemble):
        if len(X_train_full) > n_context:
            X_train, _, y_train, _ = train_test_split(X_train_full, y_train_full, train_size=n_context, random_state=seed + run_idx)
        else:
            X_train, y_train = X_train_full, y_train_full
        chunks = []
        for start in range(0, len(X_test), test_chunk_size):
            pred = reg.predict(X_train, y_train, X_test[start:start + test_chunk_size], task_type="Regression")
            if isinstance(pred, torch.Tensor):
                pred = pred.detach().float().cpu().numpy()
            chunks.append(np.asarray(pred).reshape(-1))
        pred_runs.append(np.concatenate(chunks))
        print(f"  embedding+limix run {run_idx + 1}/{n_ensemble} done ({metric.upper()}={err(pred_runs[-1], y_test):.6f})")

    metrics = summarize_regression(np.stack(pred_runs), y_test, metric)
    metrics.update({
        "n_train_context_used": int(n_context), "n_train_available": len(X_train_full), "n_test": len(X_test),
        "inference_config": str(REG_INFERENCE_CONFIG),
    })
    return metrics


# --------------------------------------------------------------------------------------------------
# embeddings + raw LimiX classification ICL (no pooler, no graphs)
# --------------------------------------------------------------------------------------------------

def run_embedding_limix_cls(
    device: torch.device, backbone_checkpoint: Path,
    X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray, seed: int = 0,
) -> dict:
    """Pretrained embeddings as plain tabular features -> raw adapter-free LimiX classification
    retrieval ICL (train = context, valid++test = query)."""
    from vendor.limix.inference.predictor import LimiXPredictor

    print(f"  Loading LimiXPredictor (cls retrieval) -- backbone {backbone_checkpoint}, config {CLS_INFERENCE_CONFIG}, seed={seed}")
    clf = LimiXPredictor(device=device, model_path=str(backbone_checkpoint), inference_config=str(CLS_INFERENCE_CONFIG), seed=seed)

    n_valid = len(X_valid)
    X_query = np.concatenate([X_valid, X_test], axis=0)
    y_query = np.concatenate([y_valid, y_test], axis=0)
    proba = clf.predict(X_train, y_train, X_query, task_type="Classification")
    if isinstance(proba, torch.Tensor):
        proba = proba.detach().float().cpu().numpy()
    pos = np.asarray(proba, dtype=np.float64)[:, 1]

    return {
        **three_way_metrics(pos, y_query, n_valid),
        "n_train_context": int(len(X_train)),
        "inference_config": str(CLS_INFERENCE_CONFIG),
    }


# --------------------------------------------------------------------------------------------------
# shared classification driver
# --------------------------------------------------------------------------------------------------

PIPELINE_KEYS = [
    "baseline_extra_features_only",
    "extra_features_only_molebert_augmented",
    "baseline_x_plus_extra_features",
    "x_plus_extra_features_molebert_augmented",
    "molebert_limix",
]
PIPELINE_TITLES = {
    "baseline_extra_features_only": "1. baseline_extra_features_only (extra RDKit atom features alone, no data.x)",
    "extra_features_only_molebert_augmented": "2. extra_features_only_molebert_augmented (+ embedding feature group spliced into the pooler)",
    "baseline_x_plus_extra_features": "3. baseline_x_plus_extra_features (data.x + extra RDKit atom features)",
    "x_plus_extra_features_molebert_augmented": "4. x_plus_extra_features_molebert_augmented (+ embedding feature group spliced into the pooler)",
    "molebert_limix": "5. molebert_limix (embeddings as tabular features -> raw LimiX cls ICL, no pooler/graphs)",
}


def run_classification(args: argparse.Namespace, dataset: str, tasks: list[int], task_names: list[str]) -> dict:
    """Runs all five pipelines on one MoleculeNet classification dataset, once per task column in
    `tasks` (labels from PyG data.y). Context = scaffold train split, query = valid ++ test.
    Returns {pipeline_key: {task_name: {valid|test|combined: metrics}} or None} plus bookkeeping."""
    device = torch.device(args.device)
    model = load_backbone(args.backbone_checkpoint, device)
    pooler, ckpt_info = load_pooler(args.pooler_checkpoint, model, device, use_ema=not args.no_ema)

    _, meta, ex, emb, emb_dir = load_all_splits(args, dataset)
    n_context, n_valid, n_test = len(ex["train"]), len(ex["valid"]), len(ex["test"])
    examples = ex["train"] + ex["valid"] + ex["test"]
    labels = {s: labels_matrix(ex[s]) for s in SPLITS}
    labels_all = np.concatenate([labels[s] for s in SPLITS], axis=0)
    if np.isnan(labels_all[:, tasks]).any():
        raise ValueError(f"{dataset}: missing (NaN) labels in the selected task columns -- not supported here.")
    n_zero_edge = sum(1 for d in examples if d.edge_index.numel() == 0)
    print(f"  context(train)={n_context}, query valid={n_valid}, query test={n_test}; "
          f"{n_zero_edge} zero-bond molecule(s) self-looped; tasks={task_names}; "
          f"{args.limix_n_members} permutation member(s), --limix-seed={args.limix_seed}")

    # Label-independent encodings, built once and shared by every task (labels swapped via y_override).
    placeholder_y = torch.zeros(len(examples))
    datasets = {
        variant: build_permuted_datasets(
            model, device, examples, featurize(examples, variant), placeholder_y,
            n_context, args.limix_n_members, args.limix_seed,
        )
        for variant in FEATURE_VARIANTS
    }

    emb_grouped = None
    if not args.skip_embeddings:
        emb_all = torch.from_numpy(np.concatenate([emb[s] for s in SPLITS], axis=0)).float().to(device)
        with torch.no_grad(), autocast_ctx(device):
            emb_grouped = encode_embeddings_as_groups(model, emb_all, eval_pos_molecules=n_context)
    run_limix = emb_grouped is not None and not args.skip_embedding_limix
    if run_limix and device.type != "cuda":
        print("  [warn] embedding+raw-LimiX pipeline needs CUDA -- skipping it.")
        run_limix = False

    results: dict[str, dict | None] = {k: {} for k in PIPELINE_KEYS}
    for t, name in zip(tasks, task_names):
        task_y = torch.tensor(labels_all[:, t], dtype=torch.float32, device=device)
        for variant, base_key, aug_key in (
            ("extra_features_only", "baseline_extra_features_only", "extra_features_only_molebert_augmented"),
            ("x_plus_extra_features", "baseline_x_plus_extra_features", "x_plus_extra_features_molebert_augmented"),
        ):
            pos, target = run_pooler_ensemble_cls(model, pooler, device, datasets[variant], args.limix_seed, y_override=task_y)
            results[base_key][name] = three_way_metrics(pos, target, n_valid)
            if emb_grouped is not None:
                pos, target = run_pooler_ensemble_cls(
                    model, pooler, device, datasets[variant], args.limix_seed,
                    embeddings_grouped=emb_grouped, y_override=task_y,
                )
                results[aug_key][name] = three_way_metrics(pos, target, n_valid)
        if run_limix:
            print(f"\n{args.embedding_model} + raw LimiX classification ICL -- task '{name}'...")
            results["molebert_limix"][name] = run_embedding_limix_cls(
                device, args.backbone_checkpoint,
                emb["train"], labels["train"][:, t], emb["valid"], labels["valid"][:, t],
                emb["test"], labels["test"][:, t], seed=args.limix_seed,
            )
        base = results["baseline_x_plus_extra_features"][name]
        print(f"  task {name}: x_plus_extra test roc_auc={base['test']['roc_auc']:.4f}")

    for k in PIPELINE_KEYS:
        if not results[k]:
            results[k] = None

    return {
        "results": results,
        "ckpt_info": ckpt_info,
        "meta_provenance": {k: meta.get(k) for k in ("model", "encoder", "checkpoint", "pooling", "split")},
        "embeddings_dir": str(emb_dir),
        "n_context": n_context, "n_valid": n_valid, "n_test": n_test,
        "n_zero_edge_molecules_kept": n_zero_edge,
        "labels_all": labels_all,
    }


def common_output_fields(args: argparse.Namespace, run: dict) -> dict:
    from atom_features import EXTRA_FEATURE_NAMES, X_COLUMNS, X_PLUS_EXTRA_FEATURE_COLUMNS

    return {
        "pooler_checkpoint": str(args.pooler_checkpoint),
        "pooler_ckpt_info": run["ckpt_info"],
        "backbone_checkpoint": str(args.backbone_checkpoint),
        "embedding_model": args.embedding_model,
        "embeddings_dir": run["embeddings_dir"],
        "embedding_meta": run["meta_provenance"],
        "extra_feature_columns": EXTRA_FEATURE_NAMES,
        "x_columns": X_COLUMNS,
        "x_plus_extra_feature_columns": X_PLUS_EXTRA_FEATURE_COLUMNS,
        "isolated_atom_handling": "self_loop",
        "context": "scaffold_split_train_full",
        "query": "scaffold_split_valid_plus_test",
        "n_context": run["n_context"], "n_valid": run["n_valid"], "n_test": run["n_test"],
        "n_zero_edge_molecules_kept": run["n_zero_edge_molecules_kept"],
        "limix_seed": args.limix_seed,
        "limix_n_members": args.limix_n_members,
    }


def print_cls_block(title: str, per_task: dict | None, task_names: list[str], average: dict | None = None) -> None:
    print(f"\n-- {title} --")
    if per_task is None:
        print("  (skipped)")
        return
    print(f"{'split':10s} {'task':16s} {'roc_auc':>10s} {'ap':>10s} {'accuracy':>10s}")
    for split in ("valid", "test", "combined"):
        for name in task_names:
            m = per_task[name][split]
            print(f"{split:10s} {name:16s} {m['roc_auc']:10.4f} {m['ap']:10.4f} {m['accuracy']:10.4f}")
        if average is not None:
            a = average[split]
            vals = [a[k]["mean"] if isinstance(a[k], dict) else a[k] for k in ("roc_auc", "ap", "accuracy")]
            print(f"{split:10s} {'AVG':16s} {vals[0]:10.4f} {vals[1]:10.4f} {vals[2]:10.4f}")
