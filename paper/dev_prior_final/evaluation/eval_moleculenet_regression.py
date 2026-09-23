"""Pooler checkpoint -> MoleculeNet REGRESSION (ESOL / FreeSolv / Lipophilicity), scaffold split:
context resampled from train, query = test. EVAL-ONLY.

FIVE PIPELINES (same structure as the classification scripts, see common):
  1. baseline_extra_features_only            -- atom features = 31 RDKit columns alone.
  2. extra_features_only_molebert_augmented  -- 1 + the pretrained molecule embedding (--embedding-model)
                                                spliced into the pooler as one more LimiX feature group.
  3. baseline_x_plus_extra_features           -- atom features = [data.x (9) | 31 RDKit columns].
  4. x_plus_extra_features_molebert_augmented -- 3 + the same embedding splice.
  5. molebert_limix                           -- embeddings as plain tabular features -> raw LimiX
                                                regression retrieval ICL (no pooler, no graphs).
(Output keys are named "molebert_*" for every embedding model; the model actually used is recorded under
"embedding_model".)

Two-level ensembling for pipelines 1-4 (common.run_regression_ensemble): --n-ensemble context
resamples (--max-train molecules drawn without replacement from train, seeded by --seed) x
--limix-n-members seeded feature-column permutations, averaged within each run. Pipeline 5 resamples
its context the same number of times and relies on LimiXPredictor's own internal ensembling. Nothing
is dropped: zero-degree atoms get a self-loop. Metrics in real target units: RMSE and R2.

Usage:
    python eval_moleculenet_regression.py --dataset esol \\
        --moleculenet-root DIR --pooler-checkpoint PATH --embedding-model {Molbert,MolDeBERTa} \\
        [--max-train N] [--n-ensemble N] [--seed S] [--only-x-plus-extra] [--output-json PATH]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

import common as ec
from atom_features import EXTRA_FEATURE_NAMES, X_COLUMNS, X_PLUS_EXTRA_FEATURE_COLUMNS

DATASETS = {
    "esol": "ESOL (log solubility, mols/L)",
    "freesolv": "FreeSolv (hydration free energy, kcal/mol)",
    "lipo": "Lipophilicity (octanol/water logD)",
}
METRIC = "rmse"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS), help="Which MoleculeNet regression benchmark to evaluate.")
    ec.add_common_args(parser)
    parser.add_argument("--max-train", type=int, default=None, help="CONTEXT size (default: the full train split).")
    parser.add_argument("--max-test", type=int, default=None, help="QUERY size, one fixed subsample shared by every run (default: full test split).")
    parser.add_argument("--n-ensemble", type=int, default=10, help="Number of context-resampling runs (default: %(default)s).")
    parser.add_argument("--test-chunk-size", type=int, default=None,
                        help="Score the query set in batches of this size (memory only; default: all at once).")
    parser.add_argument("--seed", type=int, default=0, help="Seed for context resampling / --max-test (default: %(default)s).")
    parser.add_argument("--only-x-plus-extra", action="store_true",
                        help="Skip pipelines 1-2 (the extra_features_only variant) -- roughly halves pipelines 1-4's runtime.")
    return parser.parse_args()


def subsample_with_emb(examples: list, emb: np.ndarray, max_n: int | None, seed: int) -> tuple[list, np.ndarray]:
    """One fixed query subsample (not resampled per run), keeping the embedding rows aligned."""
    if max_n is None or max_n >= len(examples):
        return examples, emb
    sel = np.random.default_rng(seed).choice(len(examples), size=max_n, replace=False)
    return [examples[i] for i in sel], emb[sel]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset_key = args.dataset

    model = ec.load_backbone(args.backbone_checkpoint, device)
    pooler, ckpt_info = ec.load_pooler(args.pooler_checkpoint, model, device, use_ema=not args.no_ema)

    _, meta, ex, emb, emb_dir = ec.load_all_splits(args, dataset_key)
    train_examples, train_emb_np = ex["train"], emb["train"]
    test_examples, test_emb_np = subsample_with_emb(ex["test"], emb["test"], args.max_test, args.seed)
    max_train = args.max_train or len(train_examples)
    test_chunk_size = args.test_chunk_size or len(test_examples)

    train_y = torch.from_numpy(ec.labels_matrix(train_examples)[:, 0]).float()
    test_y = torch.from_numpy(ec.labels_matrix(test_examples)[:, 0]).float()
    y_mean, y_std = train_y.mean().item(), max(train_y.std().item(), 1e-6)
    print(f"{dataset_key}: train={len(train_examples)}, test={len(test_examples)} ({DATASETS[dataset_key]}); "
          f"train target mean={y_mean:.6f}, std={y_std:.6f}; max_train={max_train}, n_ensemble={args.n_ensemble}, "
          f"limix_n_members={args.limix_n_members}, limix_seed={args.limix_seed}")

    train_emb = None if args.skip_embeddings else torch.from_numpy(train_emb_np).float()
    test_emb = None if args.skip_embeddings else torch.from_numpy(test_emb_np).float()

    variants = [("x_plus_extra_features", "baseline_x_plus_extra_features", "x_plus_extra_features_molebert_augmented")]
    if not args.only_x_plus_extra:
        variants.insert(0, ("extra_features_only", "baseline_extra_features_only", "extra_features_only_molebert_augmented"))

    result = {
        "dataset": dataset_key, "target": DATASETS[dataset_key], "isolated_atom_policy": "self_loop_keepall",
        "pooler_checkpoint": str(args.pooler_checkpoint), "pooler_ckpt_info": ckpt_info,
        "backbone_checkpoint": str(args.backbone_checkpoint),
        "embedding_model": args.embedding_model, "embeddings_dir": str(emb_dir),
        "embedding_meta": {k: meta.get(k) for k in ("model", "encoder", "checkpoint", "pooling", "split")},
        "context": "scaffold_split_train_resampled", "query": "scaffold_split_test",
        "max_train": max_train, "n_ensemble": args.n_ensemble, "seed": args.seed, "test_chunk_size": test_chunk_size,
        "limix_seed": args.limix_seed, "limix_n_members": args.limix_n_members,
        "n_train_available": len(train_examples), "n_test": len(test_examples),
        "extra_feature_columns": EXTRA_FEATURE_NAMES, "x_columns": X_COLUMNS,
        "x_plus_extra_feature_columns": X_PLUS_EXTRA_FEATURE_COLUMNS,
    }

    target_ref = None
    for variant, base_key, aug_key in variants:
        print(f"\nRunning {args.n_ensemble} run(s) x {args.limix_n_members} member(s) [{variant}]...")
        runs_base, runs_aug, target = ec.run_regression_ensemble(
            model, pooler, device, train_examples, test_examples,
            ec.featurize(train_examples, variant), ec.featurize(test_examples, variant),
            train_y, test_y, train_emb, test_emb, y_mean, y_std, max_train, args.n_ensemble, args.seed,
            test_chunk_size, args.limix_seed, args.limix_n_members, metric=METRIC,
        )
        if target_ref is not None:
            assert np.array_equal(target_ref, target), "the two atom-feature variants scored a different query set"
        target_ref = target
        result[base_key] = ec.summarize_regression(runs_base, target, METRIC)
        result[aug_key] = ec.summarize_regression(runs_aug, target, METRIC) if runs_aug is not None else None

    if not args.skip_embeddings and not args.skip_embedding_limix:
        if device.type != "cuda":
            raise RuntimeError("The embedding+raw-LimiX pipeline requires CUDA; pass --skip-embedding-limix to run on CPU.")
        print(f"\n=== 5. {args.embedding_model} embeddings + raw LimiX tabular ICL ===")
        result["molebert_limix"] = ec.run_embedding_limix_reg(
            device, args.backbone_checkpoint, train_emb_np, train_y.numpy().astype(np.float64),
            test_emb_np, test_y.numpy().astype(np.float64), max_train, args.n_ensemble, args.seed,
            test_chunk_size, args.limix_seed, metric=METRIC,
        )

    print(f"\n=== {dataset_key} self-loop ICL eval ({args.embedding_model} embeddings) ===")
    for key in ec.PIPELINE_KEYS:
        m = result.get(key)
        if m is not None:
            print(f"-- {ec.PIPELINE_TITLES[key].split(' (')[0]}: ensemble RMSE={m['ensemble_rmse']:.6f}, R2={m['ensemble_r2']:.6f}")

    output_json = ec.resolve_output_json(args, f"eval_{dataset_key}")
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {output_json}")


if __name__ == "__main__":
    main()
