"""Embeddings-only k-nearest-neighbours baseline: pretrained molecule embeddings (--embedding-model) as
plain tabular features -> sklearn KNeighbors{Classifier,Regressor}. No pooler, no graphs. CPU-only.

The KNN counterpart of TabICL/eval_tabicl.py. It reuses common's loaders, so it sees exactly the same
molecules, labels and splits:
  * classification (bace / bbbp / clintox / sider): fit set = the full scaffold train split; valid and
    test are scored. ROC-AUC / AP / accuracy for valid / test / combined, plus the 2-task average for
    ClinTox and delta-AUPRC + the across-task average for SIDER.
  * regression (esol / freesolv / lipo / zinc / aqsol): fit set = the full train split; valid and test
    are scored with RMSE / MAE / R2. ZINC and AQSOL use their official PyG train/val/test splits.

Model selection: for every task, each (distance, k) in --distances x --k-grid is fit on train and
scored on valid; the one with the best valid --select-by score is "selected" and its test metrics are
the reported result. Default --select-by: roc_auc for classification, rmse for esol / freesolv / lipo,
mae for zinc / aqsol.

Pipeline: [StandardScaler, fit on train] -> [L2 Normalizer if distance == cosine] -> KNN. Cosine
distance is euclidean on L2-normalised vectors.

Usage:
    python eval_knn.py --dataset bace --embedding-model Molbert
    python eval_knn.py --dataset sider --embedding-model MolDeBERTa --tasks all
    python eval_knn.py --dataset all --embedding-model Molbert --weights distance
    python eval_knn.py --dataset zinc --embedding-model Molbert --select-by mae --output-json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from torch_geometric.datasets import AQSOL, ZINC

KNN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(KNN_DIR.parent))  # evaluation/: common, eval_clintox, eval_sider

import common as ec  # noqa: E402
import eval_clintox  # noqa: E402
import eval_sider  # noqa: E402

DEFAULT_OUTPUT_ROOT = KNN_DIR / "outputs"
DEFAULT_K_GRID = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]
DISTANCES = ["euclidean", "cosine", "manhattan"]

# Classification datasets -> task names (column order of PyG's data.y). SIDER's come from --tasks.
CLS_DATASETS = {"bace": ["Class"], "bbbp": ["p_np"], "clintox": eval_clintox.TASKS, "sider": None}
# Regression datasets -> dataset root kind ("moleculenet" = scaffold split via common.load_all_splits).
REG_DATASETS = {"esol": "moleculenet", "freesolv": "moleculenet", "lipo": "moleculenet",
                "zinc": "zinc", "aqsol": "aqsol"}
ALL_DATASETS = [*CLS_DATASETS, *REG_DATASETS]

CLS_METRICS = ["roc_auc", "ap", "accuracy"]
REG_METRICS = ["rmse", "mae", "r2"]
HIGHER_IS_BETTER = {"roc_auc": True, "ap": True, "accuracy": True, "rmse": False, "mae": False, "r2": True}
DEFAULT_SELECT_BY = {**{d: "roc_auc" for d in CLS_DATASETS}, **{d: "rmse" for d in REG_DATASETS},
                     "zinc": "mae", "aqsol": "mae"}

# PyG's ZINC / AQSOL call the validation split "val"; the embeddings (and common.SPLITS) call it "valid".
PYG_SPLIT_NAME = {"train": "train", "valid": "val", "test": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", nargs="+", required=True, choices=[*ALL_DATASETS, "all"],
                        help="One or more datasets, or 'all'.")
    parser.add_argument("--embedding-model", choices=ec.EMBEDDING_MODELS, required=True,
                        help="Which pretrained molecule embeddings to use as KNN features.")
    parser.add_argument("--embeddings-root", type=Path, default=ec.DEFAULT_EMBEDDINGS_ROOT,
                        help="Root holding <embedding-model>/<dataset>/ (default: %(default)s).")
    for root in ("moleculenet", "zinc", "aqsol"):
        parser.add_argument(f"--{root}-root", type=Path, default=ec.DEFAULT_DATASETS_ROOT / root,
                            help=ec.DATASET_ROOT_HELP[root] + " (default: %(default)s)")
    parser.add_argument("--tasks", type=eval_sider._parse_tasks, default=eval_sider.DEFAULT_TEST_TASKS,
                        help=f"SIDER only: 'all', a range like '21-26', or a comma list (default {eval_sider.DEFAULT_TEST_TASKS}).")
    parser.add_argument("--k-grid", type=int, nargs="+", default=DEFAULT_K_GRID,
                        help="k values to sweep (default: %(default)s).")
    parser.add_argument("--distances", nargs="+", choices=DISTANCES, default=["euclidean", "cosine"],
                        help="Distances to sweep (default: %(default)s).")
    parser.add_argument("--weights", choices=["uniform", "distance"], default="uniform")
    parser.add_argument("--no-standardize", dest="standardize", action="store_false",
                        help="Skip the per-feature StandardScaler (fit on train; on by default).")
    parser.add_argument("--select-by", choices=[*CLS_METRICS, *REG_METRICS], default=None,
                        help="Valid metric used to pick (distance, k) (default: roc_auc for classification, "
                        "rmse for esol/freesolv/lipo, mae for zinc/aqsol). Must match the task type of every --dataset.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help="Results root; each dataset goes to <output-root>/<embedding-model>/knn_<dataset>.json "
                        "(default: %(default)s).")
    parser.add_argument("--output-json", type=Path, default=None,
                        help="Explicit metrics JSON path (only with a single --dataset; overrides --output-root).")
    args = parser.parse_args()

    if "all" in args.dataset:
        args.dataset = ALL_DATASETS
    args.dataset = list(dict.fromkeys(args.dataset))
    if args.output_json is not None and len(args.dataset) > 1:
        parser.error("--output-json needs a single --dataset; use --output-root for several.")
    if args.select_by is not None:
        allowed = CLS_METRICS if args.select_by in CLS_METRICS else REG_METRICS
        bad = [d for d in args.dataset if (d in CLS_DATASETS) != (allowed is CLS_METRICS)]
        if bad:
            parser.error(f"--select-by {args.select_by} does not apply to {bad}.")
    return args


# --------------------------------------------------------------------------------------------------
# KNN model + selection
# --------------------------------------------------------------------------------------------------

def build_knn(estimator_cls, k: int, distance: str, weights: str, standardize: bool):
    steps = [StandardScaler()] if standardize else []
    if distance == "cosine":
        steps.append(Normalizer(norm="l2"))
        distance = "euclidean"
    steps.append(estimator_cls(n_neighbors=k, weights=weights, metric=distance))
    return make_pipeline(*steps)


def _selection_key(entry: dict, sel: str) -> float:
    """Valid score oriented so larger is better; NaN (e.g. single-class valid slice) ranks last."""
    v = entry["valid"][sel]
    if not np.isfinite(v):
        return -np.inf
    return v if HIGHER_IS_BETTER[sel] else -v


def sweep_and_select(args: argparse.Namespace, k_grid: list[int], score_fn, sel: str) -> tuple[dict, dict]:
    """score_fn(distance, k) -> {split: metrics}. Returns (selected entry, {distance: [entries]})."""
    sweep = {d: [{"k": k, **score_fn(d, k)} for k in k_grid] for d in args.distances}
    best_distance, best = max(
        ((d, e) for d, entries in sweep.items() for e in entries), key=lambda de: _selection_key(de[1], sel),
    )
    selected = {"selection_criterion": f"valid {sel}", "distance": best_distance, **best}
    return selected, sweep


def config_block(args: argparse.Namespace, k_grid: list[int], sel: str) -> dict:
    return {"k_grid": k_grid, "distances": args.distances, "weights": args.weights,
            "standardize": args.standardize, "select_by": sel}


# --------------------------------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------------------------------

def run_classification(args: argparse.Namespace, dataset: str) -> dict:
    """Per task: sweep (distance, k) fit on train, pick by valid, report valid / test / combined."""
    sel = args.select_by or DEFAULT_SELECT_BY[dataset]
    if dataset == "sider":
        tasks, names = args.tasks, [str(t) for t in args.tasks]
    else:
        names = CLS_DATASETS[dataset]
        tasks = list(range(len(names)))

    _, meta, ex, emb, emb_dir = ec.load_all_splits(args, dataset)
    labels = {s: ec.labels_matrix(ex[s]) for s in ec.SPLITS}
    n_valid = len(ex["valid"])
    X_query = np.concatenate([emb["valid"], emb["test"]], axis=0)
    labels_query = np.concatenate([labels["valid"], labels["test"]], axis=0)
    if np.isnan(labels["train"][:, tasks]).any() or np.isnan(labels_query[:, tasks]).any():
        raise ValueError(f"{dataset}: missing (NaN) labels in the selected task columns -- not supported here.")
    k_grid = [k for k in args.k_grid if k <= len(ex["train"])]
    print(f"  train={len(ex['train'])}, valid={n_valid}, test={len(ex['test'])}; "
          f"features={emb['train'].shape[1]}; tasks={names}; selecting by valid {sel}")

    per_task, sweeps = {}, {}
    for t, name in zip(tasks, names):
        y_train = labels["train"][:, t].astype(np.int64)
        y_query = labels_query[:, t].astype(np.int64)

        def score(distance: str, k: int) -> dict:
            clf = build_knn(KNeighborsClassifier, k, distance, args.weights, args.standardize)
            clf.fit(emb["train"], y_train)
            classes = list(clf.classes_)
            # A train split with no positives gives classes_ == [0]: every query gets P(pos) = 0.
            pos = clf.predict_proba(X_query)[:, classes.index(1)] if 1 in classes else np.zeros(len(X_query))
            return ec.three_way_metrics(pos, y_query, n_valid)

        per_task[name], sweeps[name] = sweep_and_select(args, k_grid, score, sel)
        s = per_task[name]
        print(f"  task {name}: selected {s['distance']} k={s['k']} (valid {sel}={s['valid'][sel]:.4f}) "
              f"-> test roc_auc={s['test']['roc_auc']:.4f} ap={s['test']['ap']:.4f}")

    average = None
    if dataset == "clintox":
        average = eval_clintox._avg_over_tasks(per_task)
        knn = {"tasks": per_task, "average": average}
    elif dataset == "sider":
        # delta-AUPRC = AP - positive rate of the same query slice.
        slices = {"valid": labels["valid"], "test": labels["test"], "combined": labels_query}
        for t, name in zip(tasks, names):
            for split, lab in slices.items():
                per_task[name][split]["delta_auprc"] = per_task[name][split]["ap"] - float(lab[:, t].mean())
        average = eval_sider._avg_over_tasks(per_task, names)
        knn = {"tasks": per_task, "average": average}
    else:
        knn = per_task[names[0]]

    ec.print_cls_block(f"{args.embedding_model} embeddings -> KNN classification (selected by valid {sel})",
                       per_task, names, average)
    return {
        "task_type": "classification",
        "targets": names,
        "embeddings_dir": str(emb_dir),
        "embedding_meta": {k: meta.get(k) for k in ("model", "encoder", "checkpoint", "pooling", "split")},
        "context": "scaffold_split_train_full", "query": "scaffold_split_valid_plus_test",
        "n_train": len(ex["train"]), "n_valid": n_valid, "n_test": len(ex["test"]),
        "config": config_block(args, k_grid, sel),
        "knn": knn,
        "sweep": sweeps if len(names) > 1 else sweeps[names[0]],
    }


# --------------------------------------------------------------------------------------------------
# regression
# --------------------------------------------------------------------------------------------------

def load_regression_splits(args: argparse.Namespace, dataset: str) -> tuple[dict, dict, Path]:
    """({train|valid|test: (X, y)}, meta, embeddings_dir) over the full splits."""
    root_kind = REG_DATASETS[dataset]
    if root_kind == "moleculenet":
        _, meta, ex, emb, emb_dir = ec.load_all_splits(args, dataset)
        return {s: (emb[s], ec.labels_matrix(ex[s])[:, 0]) for s in ec.SPLITS}, meta, emb_dir

    emb_dir = ec.embeddings_dir_for(args, dataset)
    meta = ec.load_meta(emb_dir, dataset)
    data = {}
    for split in ec.SPLITS:
        if root_kind == "zinc":
            ds = ZINC(root=str(args.zinc_root), subset=True, split=PYG_SPLIT_NAME[split])
        else:
            ds = AQSOL(root=str(args.aqsol_root), split=PYG_SPLIT_NAME[split])
        examples, emb, _ = ec.load_local_index_split(ds, meta, emb_dir, dataset, split)
        data[split] = (emb, ec.labels_matrix(examples)[:, 0])
    print(f"  {args.embedding_model} embeddings from {emb_dir}: "
          + ", ".join(f"{s}={data[s][0].shape}" for s in ec.SPLITS))
    return data, meta, emb_dir


def reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_regression(args: argparse.Namespace, dataset: str) -> dict:
    """Sweep (distance, k) fit on train, pick by valid, report valid / test."""
    sel = args.select_by or DEFAULT_SELECT_BY[dataset]
    data, meta, emb_dir = load_regression_splits(args, dataset)
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = (data[s] for s in ec.SPLITS)
    k_grid = [k for k in args.k_grid if k <= len(X_train)]
    print(f"  train={len(X_train)}, valid={len(X_valid)}, test={len(X_test)}; features={X_train.shape[1]}; "
          f"target mean/std (train)={y_train.mean():.3f}/{y_train.std():.3f}; selecting by valid {sel}")

    def score(distance: str, k: int) -> dict:
        reg = build_knn(KNeighborsRegressor, k, distance, args.weights, args.standardize)
        reg.fit(X_train, y_train)
        return {"valid": reg_metrics(y_valid, reg.predict(X_valid)), "test": reg_metrics(y_test, reg.predict(X_test))}

    selected, sweep = sweep_and_select(args, k_grid, score, sel)
    print(f"\n-- {args.embedding_model} embeddings -> KNN regression: selected {selected['distance']} "
          f"k={selected['k']} (valid {sel}={selected['valid'][sel]:.4f}) --")
    print(f"{'split':10s} {'rmse':>10s} {'mae':>10s} {'r2':>10s}")
    for split in ("valid", "test"):
        m = selected[split]
        print(f"{split:10s} {m['rmse']:10.4f} {m['mae']:10.4f} {m['r2']:10.4f}")
    return {
        "task_type": "regression",
        "embeddings_dir": str(emb_dir),
        "embedding_meta": {k: meta.get(k) for k in ("model", "encoder", "checkpoint", "pooling", "split")},
        "n_train": len(X_train), "n_valid": len(X_valid), "n_test": len(X_test),
        "config": config_block(args, k_grid, sel),
        "knn": selected,
        "sweep": sweep,
    }


def main() -> None:
    args = parse_args()
    for dataset in args.dataset:
        print(f"\n=== KNN on {dataset} ({args.embedding_model} embeddings) ===")
        run = run_classification(args, dataset) if dataset in CLS_DATASETS else run_regression(args, dataset)

        output_json = args.output_json or (args.output_root / args.embedding_model / f"knn_{dataset}.json")
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump({"dataset": dataset, "model": "KNN", "embedding_model": args.embedding_model, **run}, f, indent=2)
        print(f"\nSaved to {output_json}")


if __name__ == "__main__":
    main()
