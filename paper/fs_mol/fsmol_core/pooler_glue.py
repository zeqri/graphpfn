"""Glue between the FS-Mol trainers/evaluators in this directory and the pooler code in
../dev_prior_final/.

The FS-Mol scripts use three helper namespaces, called `bt` (pooler architecture + training
helpers), `ef` (extra per-molecule feature-group forward pass) and `afe` (RDKit atom features).
They are backed by ../dev_prior_final/:

    bt   -> dev_prior_final/train_pooler.py        PoolingGNN, hyper-parameters, load_model
            dev_prior_final/evaluation/common.py   encode_raw_dataset_on_gpu, forward_pass_cls
    ef   -> dev_prior_final/evaluation/common.py   pooling_gnn_forward, encode_embeddings_as_groups
    afe  -> dev_prior_final/evaluation/atom_features.py

Only what has no counterpart there is written here: the FS-Mol checkpoint schema (`save_checkpoint`
stores `best_held_out_ap`, the key the FS-Mol trainers/evaluators read back), `_load_pooler_module`,
and the default checkpoint paths.

Function names that still say "molebert" (`ef.encode_molebert_as_groups`,
`ef.forward_pass_with_molebert`, ...) are legacy names for "encode one extra per-molecule vector as
extra LimiX feature groups and concatenate it onto `pooled`". In the FS-Mol scripts that vector is
the ECFP fingerprint (or nothing, when `None` is passed). No Mole-BERT / MolDeBERTa model or
embedding is involved.

Usage: `ns = pooler_glue.load(paper_dir)` -> ns.bt, ns.ef, ns.afe (and ns.atom_features, the whole
 dev_prior_final/evaluation/atom_features.py module).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

GLUE_DIR = Path(__file__).resolve().parent
PAPER_DIR_DEFAULT = GLUE_DIR.parent.parent              # fsmol_core/ -> fs_mol/ -> paper/
ROOT_DIR_DEFAULT = PAPER_DIR_DEFAULT.parent             # paper/ -> repo root (checkpoints/, datasets/)

# Default warm start of every FS-Mol trainer / evaluator (override with --pooler-checkpoint).
POOLER_CHECKPOINT_DEFAULT = ROOT_DIR_DEFAULT / "checkpoints" / "pooler_checkpoint_best.pt"
# Frozen LimiX-16M backbone.
BACKBONE_CHECKPOINT_DEFAULT = PAPER_DIR_DEFAULT / "checkpoints" / "LimiX-16M.ckpt"

_loaded: SimpleNamespace | None = None


def _autocast_ctx(device: torch.device):
    """torch.autocast(bfloat16) over the given device."""
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def save_checkpoint(
    path: Path, step: int, pooler_without_ddp: nn.Module, pooler_ema: nn.Module,
    optimizer: torch.optim.Optimizer, lr_scheduler, best_held_out_ap: float, best_step: int,
) -> None:
    """FS-Mol checkpoint schema {"step","pooler","pooler_ema","optimizer","lr_scheduler",
    "best_held_out_ap","best_step"}. (`best_held_out_ap` holds the best Dvalid mean delta-AUPRC.)
    Written to a .tmp path and renamed into place so a crash mid-write never leaves a corrupt file."""
    tmp_path = path.with_suffix(".tmp")
    torch.save(
        {
            "step": step,
            "pooler": pooler_without_ddp.state_dict(),
            "pooler_ema": pooler_ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "best_held_out_ap": best_held_out_ap,
            "best_step": best_step,
        },
        tmp_path,
    )
    tmp_path.replace(path)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device)


def load(paper_dir: Path | None = None) -> SimpleNamespace:
    """Import the dev_prior_final modules from `paper_dir` and return the `bt` / `ef` / `afe`
    namespaces the FS-Mol scripts use. Puts `paper_dir` on sys.path so `lib` / `vendor` import."""
    global _loaded
    paper_dir = Path(paper_dir or PAPER_DIR_DEFAULT).resolve()
    if _loaded is not None:
        if _loaded.paper_dir != paper_dir:
            raise RuntimeError(f"pooler_glue already loaded from {_loaded.paper_dir}, not {paper_dir}")
        return _loaded

    prior_dir = paper_dir / "dev_prior_final"
    eval_dir = prior_dir / "evaluation"
    for required in (prior_dir / "train_pooler.py", eval_dir / "common.py", eval_dir / "atom_features.py"):
        if not required.exists():
            raise SystemExit(f"expected {required} -- wrong paper dir? (must contain dev_prior_final/)")
    for p in (str(eval_dir), str(prior_dir), str(paper_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    import atom_features as _afe          # dev_prior_final/evaluation/atom_features.py
    import common as _common              # dev_prior_final/evaluation/common.py
    import train_pooler as _tp            # dev_prior_final/train_pooler.py

    def _load_pooler_module(checkpoint: dict, model: nn.Module, device: torch.device, use_ema: bool) -> nn.Module:
        """Rebuild the exact PoolingGNN the checkpoint was trained with and load its weights."""
        pooler = _tp.PoolingGNN(
            embed_dim=model.embed_dim, n_layers=_tp.N_GNN_LAYERS, dropout=_tp.DROPOUT,
            n_transformer_blocks=model.nlayers, n_heads=_tp.N_ATTN_HEADS,
        ).to(device)
        if use_ema:
            ema_multi_avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(decay=_tp.EMA_DECAY)
            pooler_ema = torch.optim.swa_utils.AveragedModel(pooler, device, multi_avg_fn=ema_multi_avg_fn)
            pooler_ema.load_state_dict(checkpoint["pooler_ema"])
            pooler = pooler_ema.module
        else:
            pooler.load_state_dict(checkpoint["pooler"], strict=True)
        pooler.eval()
        return pooler

    def encode_molebert_as_groups(model, device, features, eval_pos_molecules):
        """(n_molecules, n_features) per-molecule vector -> (n_molecules, n_groups, embed_dim)."""
        return _common.encode_embeddings_as_groups(model, features.to(device), eval_pos_molecules)

    def forward_pass_with_molebert(model, dataset, pooler_module, extra_grouped):
        """Classification forward (y_type=0); `extra_grouped` (or None) is concatenated onto `pooled`."""
        return _common.forward_pass_cls(model, dataset, pooler_module, extra_grouped)

    bt = SimpleNamespace(
        PoolingGNN=_tp.PoolingGNN,
        N_GNN_LAYERS=_tp.N_GNN_LAYERS, N_ATTN_HEADS=_tp.N_ATTN_HEADS, N_POOL_VIEWS=_tp.N_POOL_VIEWS,
        DROPOUT=_tp.DROPOUT, EMA_DECAY=_tp.EMA_DECAY,
        OPTIMIZER_TYPE=_tp.OPTIMIZER_TYPE, WEIGHT_DECAY=_tp.WEIGHT_DECAY, LR_SCHEDULER=_tp.LR_SCHEDULER,
        WARMUP_FRACTION=_tp.WARMUP_FRACTION, GRADIENT_CLIPPING_NORM=_tp.GRADIENT_CLIPPING_NORM,
        TEMPERATURE=_common.TEMPERATURE,
        CHECKPOINT_PATH=paper_dir / "checkpoints" / "LimiX-16M.ckpt",
        WARMSTART_CHECKPOINT_PATH=paper_dir.parent / "checkpoints" / "pooler_checkpoint_best.pt",
        load_model=_tp.load_model,
        encode_raw_dataset_on_gpu=_common.encode_raw_dataset_on_gpu,
        forward_pass=_common.forward_pass_cls,
        _autocast_ctx=_autocast_ctx,
        save_checkpoint=save_checkpoint,
        load_checkpoint=load_checkpoint,
    )
    ef = SimpleNamespace(
        bt=bt,
        encode_molebert_as_groups=encode_molebert_as_groups,
        forward_pass_with_molebert=forward_pass_with_molebert,
        pooling_gnn_forward_with_molebert=_common.pooling_gnn_forward,
        _load_pooler_module=_load_pooler_module,
    )
    afe = SimpleNamespace(
        EXTRA_FEATURE_NAMES=_afe.EXTRA_FEATURE_NAMES, N_EXTRA_FEATURES=_afe.N_EXTRA_FEATURES,
        safe_float=_afe.safe_float,
    )
    _loaded = SimpleNamespace(paper_dir=paper_dir, bt=bt, ef=ef, afe=afe, atom_features=_afe)
    return _loaded
