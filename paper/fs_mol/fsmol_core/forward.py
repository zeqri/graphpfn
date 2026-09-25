"""Pooler forward passes on FS-Mol episodes, shared by the trainers (with gradients) and the
evaluators (under torch.no_grad()). All heavy lifting lives in ../dev_prior_final via pooler_glue."""

from __future__ import annotations

import torch


def encode_dataset(env, model, device, raw: dict):
    """Batch a raw episode dict and run the frozen LimiX feature encoder over its atom features."""
    return env.bt.encode_raw_dataset_on_gpu(model, device, raw)


def splice_groups(env, model, device, permol, raw: dict):
    """Encode the per-molecule splice matrix (one row per molecule, context-normalised with
    eval_pos = support size) through the frozen feature encoder -> feature groups that get
    concatenated onto the pooled atom representation. Runs under no_grad internally: it is a fixed
    embedding, gradients still reach the pooler through the other input of the concatenation."""
    feats = torch.as_tensor(permol, dtype=torch.float32, device=device)
    return env.ef.encode_molebert_as_groups(model, device, feats, eval_pos_molecules=raw["eval_pos_molecules"])


def logits(env, model, dataset, pooler_module, grouped=None):
    """Classification forward -> ((n_query, 2) logits, (n_query,) target). `grouped` = splice groups or None.
    `pooler_module` must be the raw module (not a DDP wrapper)."""
    return env.ef.forward_pass_with_molebert(model, dataset, pooler_module, grouped)


def predict(env, model, dataset, pooler_module, grouped=None):
    """(P(active) numpy float array, y_true numpy int64 array) for the query molecules."""
    pred, target = logits(env, model, dataset, pooler_module, grouped)
    return (pred.softmax(dim=-1)[:, 1].float().cpu().numpy(),
            target.detach().cpu().numpy().astype("int64"))


def load_frozen_backbone(env, device):
    """The frozen LimiX-16M backbone (requires_grad=False, eval mode)."""
    print(f"Loading frozen LimiX-16M from {env.bt.CHECKPOINT_PATH}")
    model = env.bt.load_model(str(env.bt.CHECKPOINT_PATH), mask_prediction=False).to(device)
    for prm in model.parameters():
        prm.requires_grad = False
    model.eval()
    return model
