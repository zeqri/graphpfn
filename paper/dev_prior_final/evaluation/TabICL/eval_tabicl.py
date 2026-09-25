"""Embeddings-only TabICL baseline: pretrained molecule embeddings (--embedding-model) as plain tabular
features -> TabICL in-context learning (vendor/tabicl, v2 checkpoints). No pooler, no graphs. EVAL-ONLY.

The TabICL counterpart of the "molebert_limix" pipeline in the eval_*.py scripts. It reuses common's
loaders, so it sees exactly the same molecules, labels and contexts:
  * classification (bace / bbbp / clintox / sider): context = the full scaffold train split, query =
    valid ++ test, one fit + predict_proba per task. ROC-AUC / AP / accuracy for valid / test /
    combined, plus the 2-task average for ClinTox and delta-AUPRC + the across-task average for SIDER.
  * regression (esol / freesolv / lipo: RMSE; zinc / aqsol: MAE, plus R2): --n-ensemble contexts of
    --max-train molecules drawn by train_test_split(random_state=seed + run) -- the same draw as
    common.run_embedding_limix_reg -- with predictions averaged; query = the full test split. ZINC
    uses the full splits (no isolated-atom drop), as molebert_limix does.

Per-dataset defaults for --max-train / --n-ensemble mirror run_eval_all_seeds.sh.
--tabicl-seed is TabICL's random_state (its internal shuffle ensemble), the counterpart of --limix-seed.

CHECKPOINTS are NOT shipped with this repo. They are read from <paper>/checkpoints/tabicl/
(tabicl-classifier-v2-20260212.ckpt and tabicl-regressor-v2-20260212.ckpt from the Hugging Face repo
jingang/TabICL). Download them once, from the repository root:
    python -c "from huggingface_hub import hf_hub_download as d; [d('jingang/TabICL', f, local_dir='paper/checkpoints/tabicl') \\
        for f in ('tabicl-classifier-v2-20260212.ckpt', 'tabicl-regressor-v2-20260212.ckpt')]"
A missing checkpoint is downloaded there by TabICL on first use. Pass --checkpoint-dir DIR to read them
from another directory.

Usage:
    python eval_tabicl.py --dataset bace --embedding-model Molbert [--tabicl-seed 1] [--output-json PATH]
    python eval_tabicl.py --dataset sider --embedding-model MolDeBERTa --tasks all
    python eval_tabicl.py --dataset zinc --embedding-model Molbert --max-train 2000 --n-ensemble 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import AQSOL, ZINC

TABICL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TABICL_DIR.parent))  # evaluation/: common, eval_clintox, eval_sider

import common as ec  # noqa: E402  (also puts <paper> on sys.path for vendor.*)
import eval_clintox  # noqa: E402
import eval_sider  # noqa: E402
from vendor.tabicl import TabICLClassifier, TabICLRegressor  # noqa: E402

DEFAULT_OUTPUT_ROOT = TABICL_DIR / "outputs"
DEFAULT_CHECKPOINT_DIR = TABICL_DIR.parents[2] / "checkpoints" / "tabicl"   # <paper>/checkpoints/tabicl
CLS_CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"
REG_CHECKPOINT = "tabicl-regressor-v2-20260212.ckpt"

# Classification datasets -> task names (column order of PyG's data.y). SIDER's come from --tasks.
CLS_DATASETS = {"bace": ["Class"], "bbbp": ["p_np"], "clintox": eval_clintox.TASKS, "sider": None}
# Regression datasets -> (dataset root, metric, run_eval_all_seeds.sh defaults).
REG_DATASETS = {
    "esol": ("moleculenet", "rmse", {"max_train": None, "n_ensemble": 1}),
    "freesolv": ("moleculenet", "rmse", {"max_train": None, "n_ensemble": 1}),
    "lipo": ("moleculenet", "rmse", {"max_train": 2000, "n_ensemble": 10}),
    "zinc": ("zinc", "mae", {"max_train": 2000, "n_ensemble": 10}),
    "aqsol": ("aqsol", "mae", {"max_train": 2000, "n_ensemble": 10}),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=[*CLS_DATASETS, *REG_DATASETS])
    parser.add_argument("--embedding-model", choices=ec.EMBEDDING_MODELS, required=True,
                        help="Which pretrained molecule embeddings to use as TabICL's features.")
    parser.add_argument("--embeddings-root", type=Path, default=ec.DEFAULT_EMBEDDINGS_ROOT,
                        help="Root holding <embedding-model>/<dataset>/ (default: %(default)s).")
    for root in ("moleculenet", "zinc", "aqsol"):
        parser.add_argument(f"--{root}-root", type=Path, default=ec.DEFAULT_DATASETS_ROOT / root,
                            help=ec.DATASET_ROOT_HELP[root] + " (default: %(default)s)")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR,
                        help=f"Directory holding {CLS_CHECKPOINT} / {REG_CHECKPOINT} (default: %(default)s).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tabicl-seed", type=int, default=0, help="random_state passed to TabICL (default: %(default)s).")
    parser.add_argument("--n-estimators", type=int, default=8, help="TabICL internal ensemble size (default: %(default)s).")
    parser.add_argument("--batch-size", type=int, default=8, help="TabICL forward batch size (default: %(default)s).")
    parser.add_argument("--tasks", type=eval_sider._parse_tasks, default=eval_sider.DEFAULT_TEST_TASKS,
                        help=f"SIDER only: 'all', a range like '21-26', or a comma list (default {eval_sider.DEFAULT_TEST_TASKS}).")
    parser.add_argument("--max-train", type=int, default=None, help="Regression only: CONTEXT size (default: per dataset).")
    parser.add_argument("--n-ensemble", type=int, default=None, help="Regression only: context-resampling runs (default: per dataset).")
    parser.add_argument("--test-chunk-size", type=int, default=None,
                        help="Regression only: score the query set in batches of this size (default: all at once).")
    parser.add_argument("--seed", type=int, default=0, help="Regression only: seed for context resampling (default: %(default)s).")
    parser.add_argument("--output-json", type=Path, default=None,
                        help=f"Metrics JSON (default: {DEFAULT_OUTPUT_ROOT}/<embedding-model>/tabicl_<dataset>.json).")
    args = parser.parse_args()
    if args.dataset in REG_DATASETS:
        for key, value in REG_DATASETS[args.dataset][2].items():
            if getattr(args, key) is None:
                setattr(args, key, value)
    return args


def make_estimator(estimator_cls, checkpoint: str, args: argparse.Namespace):
    """TabICL estimator on <--checkpoint-dir>/<checkpoint> (downloaded there by TabICL if missing)."""
    # The vendored TabICL (2.0.1) passes the device to torch.cuda.mem_get_info, which on this torch
    # version rejects a bare "cuda" -- give it an explicit index.
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return estimator_cls(
        n_estimators=args.n_estimators, batch_size=args.batch_size, device=device,
        random_state=args.tabicl_seed, checkpoint_version=checkpoint,
        model_path=args.checkpoint_dir / checkpoint,
        allow_auto_download=True, verbose=False,
    )


# --------------------------------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------------------------------

def run_classification(args: argparse.Namespace) -> dict:
    """One TabICL fit + predict_proba per task: train = context, valid ++ test = query."""
    if args.dataset == "sider":
        tasks, names = args.tasks, [str(t) for t in args.tasks]
    else:
        names = CLS_DATASETS[args.dataset]
        tasks = list(range(len(names)))

    _, meta, ex, emb, emb_dir = ec.load_all_splits(args, args.dataset)
    labels = {s: ec.labels_matrix(ex[s]) for s in ec.SPLITS}
    n_valid = len(ex["valid"])
    X_query = np.concatenate([emb["valid"], emb["test"]], axis=0)
    labels_query = np.concatenate([labels["valid"], labels["test"]], axis=0)
    if np.isnan(labels["train"][:, tasks]).any() or np.isnan(labels_query[:, tasks]).any():
        raise ValueError(f"{args.dataset}: missing (NaN) labels in the selected task columns -- not supported here.")
    print(f"  context(train)={len(ex['train'])}, query valid={n_valid}, query test={len(ex['test'])}; "
          f"features={emb['train'].shape[1]}; tasks={names}; tabicl_seed={args.tabicl_seed}")

    per_task = {}
    for t, name in zip(tasks, names):
        clf = make_estimator(TabICLClassifier, CLS_CHECKPOINT, args)
        clf.fit(emb["train"], labels["train"][:, t].astype(np.int64))
        proba = np.asarray(clf.predict_proba(X_query), dtype=np.float64)
        pos = proba[:, list(clf.classes_).index(1)]
        per_task[name] = ec.three_way_metrics(pos, labels_query[:, t], n_valid)
        print(f"  task {name}: test roc_auc={per_task[name]['test']['roc_auc']:.4f}")

    average = None
    if args.dataset == "clintox":
        average = eval_clintox._avg_over_tasks(per_task)
        tabicl = {"tasks": per_task, "average": average}
    elif args.dataset == "sider":
        # delta-AUPRC = AP - positive rate of the same query slice.
        slices = {"valid": labels["valid"], "test": labels["test"], "combined": labels_query}
        for t, name in zip(tasks, names):
            for split, lab in slices.items():
                per_task[name][split]["delta_auprc"] = per_task[name][split]["ap"] - float(lab[:, t].mean())
        average = eval_sider._avg_over_tasks(per_task, names)
        tabicl = {"tasks": per_task, "average": average}
    else:
        tabicl = per_task[names[0]]

    ec.print_cls_block(f"{args.embedding_model} embeddings -> TabICL classification ICL (no pooler, no graphs)",
                       per_task, names, average)
    return {
        "targets": names,
        "embeddings_dir": str(emb_dir),
        "embedding_meta": {k: meta.get(k) for k in ("model", "encoder", "checkpoint", "pooling", "split")},
        "context": "scaffold_split_train_full", "query": "scaffold_split_valid_plus_test",
        "n_context": len(ex["train"]), "n_valid": n_valid, "n_test": len(ex["test"]),
        "tabicl": tabicl,
    }


# --------------------------------------------------------------------------------------------------
# regression
# --------------------------------------------------------------------------------------------------

def load_regression_splits(args: argparse.Namespace) -> tuple[dict, dict, Path]:
    """({train|test: (X, y)}, meta, embeddings_dir) over the FULL splits -- the data molebert_limix uses."""
    root_kind = REG_DATASETS[args.dataset][0]
    if root_kind == "moleculenet":
        _, meta, ex, emb, emb_dir = ec.load_all_splits(args, args.dataset)
        return {s: (emb[s], ec.labels_matrix(ex[s])[:, 0]) for s in ("train", "test")}, meta, emb_dir

    emb_dir = ec.embeddings_dir_for(args, args.dataset)
    meta = ec.load_meta(emb_dir, args.dataset)
    data = {}
    for split in ("train", "test"):
        if root_kind == "zinc":
            ds = ZINC(root=str(args.zinc_root), subset=True, split=split)
        else:
            ds = AQSOL(root=str(args.aqsol_root), split=split)
        examples, emb, _ = ec.load_local_index_split(ds, meta, emb_dir, args.dataset, split)
        data[split] = (emb, ec.labels_matrix(examples)[:, 0])
    return data, meta, emb_dir


def run_regression(args: argparse.Namespace) -> dict:
    """--n-ensemble TabICL runs, each on a fresh context subsample, predictions averaged."""
    metric = REG_DATASETS[args.dataset][1]
    err = ec.REGRESSION_METRICS[metric]
    data, meta, emb_dir = load_regression_splits(args)
    (X_train_full, y_train_full), (X_test, y_test) = data["train"], data["test"]
    n_context = min(args.max_train or len(X_train_full), len(X_train_full))
    test_chunk_size = args.test_chunk_size or len(X_test)
    print(f"  train available={len(X_train_full)}, test={len(X_test)}, features={X_train_full.shape[1]}; "
          f"context={n_context}, n_ensemble={args.n_ensemble}, seed={args.seed}, tabicl_seed={args.tabicl_seed}")

    pred_runs = []
    for run_idx in range(args.n_ensemble):
        if len(X_train_full) > n_context:
            X_train, _, y_train, _ = train_test_split(
                X_train_full, y_train_full, train_size=n_context, random_state=args.seed + run_idx,
            )
        else:
            X_train, y_train = X_train_full, y_train_full
        reg = make_estimator(TabICLRegressor, REG_CHECKPOINT, args)
        reg.fit(X_train, y_train)
        chunks = [
            np.asarray(reg.predict(X_test[start:start + test_chunk_size]), dtype=np.float64).reshape(-1)
            for start in range(0, len(X_test), test_chunk_size)
        ]
        pred_runs.append(np.concatenate(chunks))
        print(f"  run {run_idx + 1}/{args.n_ensemble} done ({metric.upper()}={err(pred_runs[-1], y_test):.6f})")

    tabicl = ec.summarize_regression(np.stack(pred_runs), y_test, metric)
    print(f"\n-- {args.embedding_model} embeddings -> TabICL regression ICL: "
          f"ensemble {metric.upper()}={tabicl[f'ensemble_{metric}']:.6f}, R2={tabicl['ensemble_r2']:.6f}")
    return {
        "metric": metric,
        "embeddings_dir": str(emb_dir),
        "embedding_meta": {k: meta.get(k) for k in ("model", "encoder", "checkpoint", "pooling", "split")},
        "max_train": args.max_train, "n_ensemble": args.n_ensemble, "seed": args.seed,
        "test_chunk_size": test_chunk_size,
        "n_train_context_used": int(n_context), "n_train_available": len(X_train_full), "n_test": len(X_test),
        "tabicl": tabicl,
    }


def main() -> None:
    args = parse_args()
    print(f"=== TabICL on {args.dataset} ({args.embedding_model} embeddings) ===")
    run = run_classification(args) if args.dataset in CLS_DATASETS else run_regression(args)

    output_json = args.output_json or (DEFAULT_OUTPUT_ROOT / args.embedding_model / f"tabicl_{args.dataset}.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump({
            "dataset": args.dataset, "model": "TabICL", "embedding_model": args.embedding_model,
            "tabicl_seed": args.tabicl_seed, "n_estimators": args.n_estimators,
            "checkpoint": str(args.checkpoint_dir),
            "checkpoint_version": CLS_CHECKPOINT if args.dataset in CLS_DATASETS else REG_CHECKPOINT,
            **run,
        }, f, indent=2)
    print(f"\nSaved to {output_json}")


if __name__ == "__main__":
    main()
