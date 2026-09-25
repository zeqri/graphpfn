"""Pooler checkpoint -> AQSOL (torch_geometric AQSOL official split) LogS regression: context
resampled from the train split, query = the test split. EVAL-ONLY.

THREE PIPELINES (the ones the --limix-seed uncertainty sweep uses):
  1. baseline_x_plus_extra_features           -- atom features = [9 MoleculeNet-style columns |
                                                31 RDKit extra columns] (atom_features).
  2. x_plus_extra_features_molebert_augmented -- 1 + the pretrained molecule embedding (--embedding-model)
                                                spliced into the pooler as one more LimiX feature group.
  3. molebert_limix                           -- the embeddings as plain tabular features -> raw LimiX
                                                regression retrieval ICL (no pooler, no graphs).

ATOM FEATURES: AQSOL graphs carry no SMILES. PRIMARY: each graph is WL-matched to its row in
<aqsol-root>/data_curated.csv and both feature blocks are computed from that SMILES, permuted into
PyG node order (atom_features.load_aqsol_x_plus_extra_from_smiles, cached to --features-cache;
100% coverage on train/test). FALLBACK: an RDKit rebuild of the graph's own topology; if that fails
too, zero-fill. Per-split source counts go to "aqsol_40dim_feature_source".

ISOLATED ATOMS: every molecule is kept; zero-degree atoms (counter-ions etc.) get a self-loop.
EMBEDDINGS: <dataset>_meta.json lists local indices into each PyG split ("split_idx", or
"graph_index" in the MolDeBERTa meta); embedding row r is AQSOL(split)[indices[r]].

Pipelines 1-2 use common.run_regression_ensemble (--n-ensemble context resamples x
--limix-n-members seeded feature permutations). Metrics in real units: MAE and R2.

Usage:
    python eval_aqsol.py \\
        --aqsol-root DIR --pooler-checkpoint PATH --embedding-model {Molbert,MolDeBERTa} \\
        [--max-train 2000] [--n-ensemble 10] [--seed 0] [--limix-seed S] \\
        [--smiles-csv PATH] [--features-cache PATH] [--output-json PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from rdkit import RDLogger
from torch_geometric.datasets import AQSOL

import common as ec
from atom_features import (
    N_X_PLUS_EXTRA_FEATURES, X_PLUS_EXTRA_FEATURE_COLUMNS, aqsol_x_plus_extra_rebuild, load_aqsol_x_plus_extra_from_smiles,
)

DATASET = "aqsol"
OUTPUT_NAME = "eval_aqsol"
METRIC = "mae"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ec.add_common_args(parser, dataset_root="aqsol")
    parser.add_argument("--smiles-csv", type=Path, default=None,
                        help="AqSolDB curated SMILES table (default: <aqsol-root>/data_curated.csv).")
    parser.add_argument("--features-cache", type=Path, default=None,
                        help="Cache of the SMILES-matched 40-dim atom features "
                        "(default: <aqsol-root>/aqsol_40dim_from_smiles_cache.pt; delete to rebuild).")
    parser.add_argument("--max-train", type=int, default=2000, help="CONTEXT size (default: %(default)s).")
    parser.add_argument("--n-ensemble", type=int, default=10, help="Number of context-resampling runs (default: %(default)s).")
    parser.add_argument("--test-chunk-size", type=int, default=None,
                        help="Score the query set in batches of this size (memory only; default: all at once).")
    parser.add_argument("--seed", type=int, default=0, help="Seed for context resampling (default: %(default)s).")
    return parser.parse_args()


def build_features(
    examples: list, indices: list[int], x40_by_orig: list, atom_vocab: list[str], bond_vocab: list[str], label: str,
) -> tuple[list[torch.Tensor], list[int], dict]:
    """[n_atoms, 40] per molecule: SMILES-matched block if available, else topology rebuild, else
    zero-fill. Returns (features, zero-filled positions, source counts)."""
    feats, failed = [], []
    src = {"smiles": 0, "rebuild": 0, "zerofill": 0}
    for k, (data, orig) in enumerate(zip(examples, indices)):
        n_atoms = data.x.shape[0]
        x40 = x40_by_orig[orig] if orig < len(x40_by_orig) else None
        tag = "smiles"
        if x40 is None:
            x40 = aqsol_x_plus_extra_rebuild(data, atom_vocab, bond_vocab)
            tag = "rebuild"
        if x40 is None or tuple(x40.shape) != (n_atoms, N_X_PLUS_EXTRA_FEATURES):
            x40, tag = torch.zeros((n_atoms, N_X_PLUS_EXTRA_FEATURES), dtype=torch.float32), "zerofill"
            failed.append(k)
        feats.append(x40.float().contiguous())
        src[tag] += 1
    print(f"  [40dim] {label}: feature source {src}")
    return feats, failed, src


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    RDLogger.DisableLog("rdApp.*")
    smiles_csv = args.smiles_csv or (args.aqsol_root / "data_curated.csv")
    features_cache = args.features_cache or (args.aqsol_root / "aqsol_40dim_from_smiles_cache.pt")

    model = ec.load_backbone(args.backbone_checkpoint, device)
    pooler, ckpt_info = ec.load_pooler(args.pooler_checkpoint, model, device, use_ema=not args.no_ema)

    emb_dir = ec.embeddings_dir_for(args, DATASET)
    meta = ec.load_meta(emb_dir, DATASET)
    ex, emb, idx = {}, {}, {}
    for split in ("train", "test"):
        ds = AQSOL(root=str(args.aqsol_root), split=split)
        ex[split], emb[split], idx[split] = ec.load_local_index_split(ds, meta, emb_dir, DATASET, split)
        if any(d.x.shape[0] == 0 for d in ex[split]):
            raise ValueError(f"AQSOL {split}: zero-atom molecule(s) present -- cannot pool them.")
        print(f"AQSOL {split}: {len(ex[split])}/{len(ds)} graphs have {args.embedding_model} embeddings")
    train_examples, test_examples = ex["train"], ex["test"]

    max_train = args.max_train
    test_chunk_size = args.test_chunk_size or len(test_examples)
    train_y = torch.from_numpy(ec.labels_matrix(train_examples)[:, 0]).float()
    test_y = torch.from_numpy(ec.labels_matrix(test_examples)[:, 0]).float()
    y_mean, y_std = train_y.mean().item(), max(train_y.std().item(), 1e-6)
    print(f"Train target stats: mean={y_mean:.6f}, std={y_std:.6f}")

    print(f"\nBuilding 40-dim atom features for AQSOL (SMILES from {smiles_csv}, cache {features_cache})...")
    x40_by_split = load_aqsol_x_plus_extra_from_smiles(args.aqsol_root, smiles_csv, features_cache, ("train", "test"))
    probe = AQSOL(root=str(args.aqsol_root), split="train")
    atom_vocab, bond_vocab = list(probe.atoms()), list(probe.bonds())
    train_feats, train_failed, train_src = build_features(train_examples, idx["train"], x40_by_split["train"], atom_vocab, bond_vocab, "train")
    test_feats, test_failed, test_src = build_features(test_examples, idx["test"], x40_by_split["test"], atom_vocab, bond_vocab, "test")

    result = {
        "dataset": "AQSOL", "target": "LogS",
        "x_plus_extra_feature_columns": X_PLUS_EXTRA_FEATURE_COLUMNS,
        "pooler_checkpoint": str(args.pooler_checkpoint), "pooler_ckpt_info": ckpt_info,
        "backbone_checkpoint": str(args.backbone_checkpoint),
        "embedding_model": args.embedding_model, "embeddings_dir": str(emb_dir),
        "embedding_meta": {k: meta.get(k) for k in ("model", "encoder", "checkpoint", "pooling", "split")},
        "isolated_atom_policy": "self_loop",
        "max_train": max_train, "n_ensemble": args.n_ensemble, "seed": args.seed, "test_chunk_size": test_chunk_size,
        "limix_seed": args.limix_seed, "limix_n_members": args.limix_n_members,
        "n_train_available": len(train_examples), "n_test": len(test_examples),
        "aqsol_40dim_feature_source": {"train": train_src, "test": test_src},
        "aqsol_40dim_reconstruction_failed": {"train": train_failed, "test": test_failed},
    }

    print(f"\nRunning {args.n_ensemble} run(s) x {args.limix_n_members} member(s) [x_plus_extra_features], "
          f"max_train={max_train}, test_chunk_size={test_chunk_size}, limix_seed={args.limix_seed}...")
    train_emb = None if args.skip_embeddings else torch.from_numpy(emb["train"]).float()
    test_emb = None if args.skip_embeddings else torch.from_numpy(emb["test"]).float()
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
        print(f"\n=== molebert_limix ({args.embedding_model} embeddings + raw LimiX tabular ICL) ===")
        result["molebert_limix"] = ec.run_embedding_limix_reg(
            device, args.backbone_checkpoint, emb["train"], train_y.numpy().astype("float64"),
            emb["test"], test_y.numpy().astype("float64"), max_train, args.n_ensemble, args.seed,
            test_chunk_size, args.limix_seed, metric=METRIC,
        )

    print(f"\n=== AQSOL ICL eval ({args.embedding_model} embeddings) ===")
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
