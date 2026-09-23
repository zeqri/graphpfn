"""Pooler checkpoint -> ZINC-12k (torch_geometric ZINC(subset=True)) penalized-logP regression:
context resampled from the train split, query = the test split. EVAL-ONLY.

THREE PIPELINES (the ones the --limix-seed uncertainty sweep uses):
  1. baseline_x_plus_extra_features           -- atom features = [9 MoleculeNet-style columns |
                                                31 RDKit extra columns] (atom_features).
  2. x_plus_extra_features_molebert_augmented -- 1 + the pretrained molecule embedding (--embedding-model)
                                                spliced into the pooler as one more LimiX feature group.
  3. molebert_limix                           -- the embeddings as plain tabular features -> raw LimiX
                                                regression retrieval ICL (no pooler, no graphs).

ATOM FEATURES: ZINC has no SMILES, so an RDKit mol is rebuilt from ZINC's own atom/bond vocab graph
(atom_features.zinc_data_to_rdkit_mol) and BOTH blocks are read off that same mol -- no SMILES
round-trip, so no atom reordering. A molecule whose rebuild fails is zero-filled and listed under
"zinc_40dim_reconstruction_failed" (none on ZINC-12k).

ISOLATED ATOMS: molecules with a zero-degree atom are dropped from the pooler pipelines (same policy
as before); molebert_limix uses the full split. EMBEDDINGS: <dataset>_meta.json's split_idx holds
local indices into each PyG split; embedding row r is ZINC(split)[split_idx[r]].

Pipelines 1-2 use common.run_regression_ensemble (--n-ensemble context resamples x
--limix-n-members seeded feature permutations). Metrics in real units: MAE and R2.

Usage:
    python eval_zinc.py \\
        --zinc-root DIR --pooler-checkpoint PATH --embedding-model {Molbert,MolDeBERTa} \\
        [--max-train 2000] [--n-ensemble 10] [--seed 0] [--limix-seed S] [--output-json PATH]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from rdkit import RDLogger
from torch_geometric.datasets import ZINC

import common as ec
from atom_features import N_X_PLUS_EXTRA_FEATURES, X_PLUS_EXTRA_FEATURE_COLUMNS, load_zinc_vocab, zinc_x_plus_extra_atom_features

DATASET = "zinc"
OUTPUT_NAME = "eval_zinc"
METRIC = "mae"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ec.add_common_args(parser, dataset_root="zinc")
    parser.add_argument("--max-train", type=int, default=2000, help="CONTEXT size (default: %(default)s).")
    parser.add_argument("--n-ensemble", type=int, default=10, help="Number of context-resampling runs (default: %(default)s).")
    parser.add_argument("--test-chunk-size", type=int, default=None,
                        help="Score the query set in batches of this size (memory only; default: all at once).")
    parser.add_argument("--seed", type=int, default=0, help="Seed for context resampling (default: %(default)s).")
    return parser.parse_args()


def build_features(examples: list, atom_vocab: list[str], bond_vocab: list[str], label: str) -> tuple[list[torch.Tensor], list[int]]:
    """[n_atoms, 40] per molecule; a failed rebuild is zero-filled and its position returned."""
    feats, failed = [], []
    for k, data in enumerate(examples):
        try:
            feats.append(zinc_x_plus_extra_atom_features(data, atom_vocab, bond_vocab))
        except Exception as exc:  # noqa: BLE001 -- defensive; not hit on ZINC-12k
            print(f"  [40dim] {label}[{k}] reconstruction failed ({exc}); zero-filling")
            feats.append(torch.zeros((data.x.shape[0], N_X_PLUS_EXTRA_FEATURES), dtype=torch.float32))
            failed.append(k)
    return feats, failed


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    RDLogger.DisableLog("rdApp.*")

    model = ec.load_backbone(args.backbone_checkpoint, device)
    pooler, ckpt_info = ec.load_pooler(args.pooler_checkpoint, model, device, use_ema=not args.no_ema)

    emb_dir = ec.embeddings_dir_for(args, DATASET)
    meta = ec.load_meta(emb_dir, DATASET)
    full, emb_full, pooler_ex, pooler_emb = {}, {}, {}, {}
    for split in ("train", "test"):
        ds = ZINC(root=str(args.zinc_root), subset=True, split=split)
        full[split], emb_full[split], _ = ec.load_local_index_split(ds, meta, emb_dir, DATASET, split)
        keep = [k for k, d in enumerate(full[split])
                if ec.isolated_atom_local_indices(d.edge_index, d.x.shape[0]).numel() == 0]
        pooler_ex[split] = [full[split][k] for k in keep]
        pooler_emb[split] = emb_full[split][keep]
    train_examples, test_examples = pooler_ex["train"], pooler_ex["test"]
    print(f"ZINC usable (no isolated atoms): train={len(train_examples)}/{len(full['train'])}, "
          f"test={len(test_examples)}/{len(full['test'])}; {args.embedding_model} embeddings from {emb_dir} "
          f"(dim {emb_full['train'].shape[1]})")

    max_train = args.max_train
    test_chunk_size = args.test_chunk_size or len(test_examples)
    train_y = torch.from_numpy(ec.labels_matrix(train_examples)[:, 0]).float()
    test_y = torch.from_numpy(ec.labels_matrix(test_examples)[:, 0]).float()
    y_mean, y_std = train_y.mean().item(), max(train_y.std().item(), 1e-6)
    print(f"Train target stats: mean={y_mean:.6f}, std={y_std:.6f}")

    print("\nRebuilding 40-dim ([9 MoleculeNet-style | 31 RDKit extra]) atom features for ZINC...")
    atom_vocab, bond_vocab = load_zinc_vocab(args.zinc_root)
    train_feats, train_failed = build_features(train_examples, atom_vocab, bond_vocab, "train")
    test_feats, test_failed = build_features(test_examples, atom_vocab, bond_vocab, "test")

    result = {
        "dataset": "ZINC", "target": "penalized_logP",
        "x_plus_extra_feature_columns": X_PLUS_EXTRA_FEATURE_COLUMNS,
        "pooler_checkpoint": str(args.pooler_checkpoint), "pooler_ckpt_info": ckpt_info,
        "backbone_checkpoint": str(args.backbone_checkpoint),
        "embedding_model": args.embedding_model, "embeddings_dir": str(emb_dir),
        "embedding_meta": {k: meta.get(k) for k in ("model", "encoder", "checkpoint", "pooling", "split")},
        "isolated_atom_policy": "drop_molecule",
        "max_train": max_train, "n_ensemble": args.n_ensemble, "seed": args.seed, "test_chunk_size": test_chunk_size,
        "limix_seed": args.limix_seed, "limix_n_members": args.limix_n_members,
        "n_train_available": len(train_examples), "n_test": len(test_examples),
        "zinc_40dim_reconstruction_failed": {"train": train_failed, "test": test_failed},
    }

    print(f"\nRunning {args.n_ensemble} run(s) x {args.limix_n_members} member(s) [x_plus_extra_features], "
          f"max_train={max_train}, test_chunk_size={test_chunk_size}, limix_seed={args.limix_seed}...")
    train_emb = None if args.skip_embeddings else torch.from_numpy(pooler_emb["train"]).float()
    test_emb = None if args.skip_embeddings else torch.from_numpy(pooler_emb["test"]).float()
    runs_base, runs_aug, target = ec.run_regression_ensemble(
        model, pooler, device, train_examples, test_examples, train_feats, test_feats,
        train_y, test_y, train_emb, test_emb, y_mean, y_std, max_train, args.n_ensemble, args.seed,
        test_chunk_size, args.limix_seed, args.limix_n_members, metric=METRIC,
    )
    result["baseline_x_plus_extra_features"] = ec.summarize_regression(runs_base, target, METRIC)
    result["x_plus_extra_features_molebert_augmented"] = (
        ec.summarize_regression(runs_aug, target, METRIC) if runs_aug is not None else None
    )

    if not args.skip_embeddings and not args.skip_embedding_limix:
        if device.type != "cuda":
            raise RuntimeError("The embedding+raw-LimiX pipeline requires CUDA; pass --skip-embedding-limix to run on CPU.")
        print(f"\n=== molebert_limix ({args.embedding_model} embeddings + raw LimiX tabular ICL, full splits) ===")
        result["molebert_limix"] = ec.run_embedding_limix_reg(
            device, args.backbone_checkpoint,
            emb_full["train"], ec.labels_matrix(full["train"])[:, 0], emb_full["test"], ec.labels_matrix(full["test"])[:, 0],
            max_train, args.n_ensemble, args.seed, args.test_chunk_size or len(full["test"]), args.limix_seed, metric=METRIC,
        )

    print(f"\n=== ZINC ICL eval ({args.embedding_model} embeddings) ===")
    for key in ("baseline_x_plus_extra_features", "x_plus_extra_features_molebert_augmented", "molebert_limix"):
        m = result.get(key)
        if m is not None:
            print(f"-- {key}: ensemble MAE={m['ensemble_mae']:.6f}, R2={m['ensemble_r2']:.6f}")

    output_json = ec.resolve_output_json(args, OUTPUT_NAME)
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {output_json}")


if __name__ == "__main__":
    main()
