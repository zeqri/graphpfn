"""Embeddings-only TabPFN-v3 baseline: pretrained molecule embeddings (--embedding-model) as plain tabular
features -> TabPFN v3 in-context learning (tabpfn package, ModelVersion.V3). No pooler, no graphs.
EVAL-ONLY. The TabPFN-v3 sibling of ../TabICL/eval_tabicl.py: same datasets, splits, labels, protocol,
metrics and output layout -- only the tabular ICL backend differs.

SELF-CONTAINED: tabpfn needs torch>=2.5, which the evaluation env (dgl / torch_geometric / LimiX) does
not have, so this file does NOT import ../common.py and needs neither dgl nor torch_geometric nor
RDKit. Run it in an env with `tabpfn` installed. It reads, with no PyG:
  * embeddings: <embeddings-root>/<model>/<dataset>/<dataset>_embeddings_{train,valid,test}.npy + _meta.json
  * MoleculeNet labels: the raw CSV row each meta split_idx entry names, parsed exactly like
    torch_geometric's MoleculeNet.process (same line parser, same label column(s), float32) -- the raw
    CSV is first checked against the sha256 in split/split_info.json (datasets/make_scaffold_splits.py).
  * ZINC / AQSOL labels: raw/{train,val,test}.pickle in torch_geometric's own processing order
    (ZINC: subset .index order, target logP_SA_cycle_normalized; AQSOL: graphs with no edges skipped),
    so meta["split_idx"] / ["graph_index"] local indices line up exactly as they do in PyG.

PROTOCOL (as eval_tabicl.py):
  * classification (bace / bbbp / clintox / sider): context = the full scaffold train split, query =
    valid ++ test, one fit + predict_proba per task. ROC-AUC / AP / accuracy for valid / test /
    combined, plus the 2-task average for ClinTox and delta-AUPRC + the across-task average for SIDER.
  * regression (esol / freesolv / lipo: RMSE; zinc / aqsol: MAE, plus R2): --n-ensemble contexts of
    --max-train molecules drawn by train_test_split(random_state=seed + run), predictions averaged;
    query = the full test split. Per-dataset defaults mirror run_eval_all_seeds.sh.
  --tabpfn-seed is TabPFN's random_state; --n-estimators defaults to 8 (not v3's "auto") to match TabICL.

CHECKPOINTS: TabPFN v3 (gated repo Prior-Labs/tabpfn_3), not shipped with this repo. They are read from
<paper>/checkpoints/tabpfnv3/ (tabpfn-v3-classifier-v3_default.ckpt / tabpfn-v3-regressor-v3_default.ckpt).
A missing checkpoint is downloaded there by tabpfn on first use, after a one-time license acceptance
(browser login; the token is stored under ~/.cache/tabpfn). Pass --checkpoint-dir DIR to read them from
another directory. If TABPFN_MODEL_CACHE_DIR is set, skrub's and matplotlib's data dirs
(SKB_DATA_DIRECTORY / MPLCONFIGDIR) default next to it -- skrub writes its data dir at import.

Usage:
    python eval_tabpfnv3.py --dataset bace --embedding-model Molbert [--tabpfn-seed 1] [--output-json PATH]
    python eval_tabpfnv3.py --dataset sider --embedding-model MolDeBERTa --tasks all
    python eval_tabpfnv3.py --dataset zinc --embedding-model Molbert --max-train 2000 --n-ensemble 10
"""

from __future__ import annotations

import os
from pathlib import Path

# skrub (imported by tabpfn) and matplotlib create their data dirs at import time -- keep them next to
# the TabPFN model cache when one is configured, instead of under $HOME.
if "TABPFN_MODEL_CACHE_DIR" in os.environ:
    _cache_parent = Path(os.environ["TABPFN_MODEL_CACHE_DIR"]).resolve().parent
    os.environ.setdefault("SKB_DATA_DIRECTORY", str(_cache_parent / "skrub_data"))
    os.environ.setdefault("MPLCONFIGDIR", str(_cache_parent / "matplotlib"))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import re  # noqa: E402

import numpy as np  # noqa: E402
import sklearn.metrics  # noqa: E402
import torch  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from tabpfn import TabPFNClassifier, TabPFNRegressor  # noqa: E402
from tabpfn.constants import ModelVersion  # noqa: E402

TABPFN_DIR = Path(__file__).resolve().parent
REPO_DIR = TABPFN_DIR.parents[3]  # TabPFNv3/ -> evaluation/ -> dev_prior_final/ -> paper/ -> <repo>
DEFAULT_DATASETS_ROOT = REPO_DIR / "datasets"
DEFAULT_EMBEDDINGS_ROOT = REPO_DIR / "embeddings"
DEFAULT_OUTPUT_ROOT = TABPFN_DIR / "outputs"
DEFAULT_CHECKPOINT_DIR = REPO_DIR / "paper" / "checkpoints" / "tabpfnv3"

EMBEDDING_MODELS = ["Molbert", "MolDeBERTa"]
SPLITS = ("train", "valid", "test")
PYG_SPLIT = {"train": "train", "valid": "val", "test": "test"}  # raw file names of ZINC / AQSOL
CLS_CHECKPOINT = "tabpfn-v3-classifier-v3_default.ckpt"
REG_CHECKPOINT = "tabpfn-v3-regressor-v3_default.ckpt"
METRIC_KEYS = ("roc_auc", "ap", "accuracy")

# torch_geometric MoleculeNet.names: raw CSV, SMILES column, label column(s).
MOLECULENET = {
    "bace": ("bace.csv", 2),
    "bbbp": ("BBBP.csv", -2),
    "clintox": ("clintox.csv", slice(1, 3)),
    "esol": ("delaney-processed.csv", -2),
    "freesolv": ("SAMPL.csv", 2),
    "lipo": ("Lipophilicity.csv", 1),
    "sider": ("sider.csv", slice(1, 28)),
}
N_SIDER_TASKS = 27
DEFAULT_SIDER_TASKS = "21-26"  # PAR / Meta-MGNN "last 6 of 27" meta-test columns.

# Classification datasets -> task names (column order of the label slice). SIDER's come from --tasks.
CLS_DATASETS = {"bace": ["Class"], "bbbp": ["p_np"], "clintox": ["FDA_APPROVED", "CT_TOX"], "sider": None}
# Regression datasets -> (metric, run_eval_all_seeds.sh defaults).
REG_DATASETS = {
    "esol": ("rmse", {"max_train": None, "n_ensemble": 1}),
    "freesolv": ("rmse", {"max_train": None, "n_ensemble": 1}),
    "lipo": ("rmse", {"max_train": 2000, "n_ensemble": 10}),
    "zinc": ("mae", {"max_train": 2000, "n_ensemble": 10}),
    "aqsol": ("mae", {"max_train": 2000, "n_ensemble": 10}),
}


def parse_tasks(spec: str) -> list[int]:
    """SIDER task columns: 'all', a range like '21-26', or a comma list -- deduplicated and sorted."""
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(N_SIDER_TASKS))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    for c in out:
        if not 0 <= c < N_SIDER_TASKS:
            raise argparse.ArgumentTypeError(f"SIDER task index {c} out of range [0, {N_SIDER_TASKS})")
    return sorted(dict.fromkeys(out))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=[*CLS_DATASETS, *REG_DATASETS])
    parser.add_argument("--embedding-model", choices=EMBEDDING_MODELS, required=True,
                        help="Which pretrained molecule embeddings to use as TabPFN's features.")
    parser.add_argument("--embeddings-root", type=Path, default=DEFAULT_EMBEDDINGS_ROOT,
                        help="Root holding <embedding-model>/<dataset>/ (default: %(default)s).")
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT,
                        help="Root holding moleculenet/, zinc/, aqsol/ (default: %(default)s).")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR,
                        help=f"Directory holding {CLS_CHECKPOINT} / {REG_CHECKPOINT} (default: %(default)s).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tabpfn-seed", type=int, default=0, help="random_state passed to TabPFN (default: %(default)s).")
    parser.add_argument("--n-estimators", type=int, default=8, help="TabPFN ensemble size (default: %(default)s).")
    parser.add_argument("--tasks", type=parse_tasks, default=DEFAULT_SIDER_TASKS,
                        help=f"SIDER only: 'all', a range like '21-26', or a comma list (default {DEFAULT_SIDER_TASKS}).")
    parser.add_argument("--max-train", type=int, default=None, help="Regression only: CONTEXT size (default: per dataset).")
    parser.add_argument("--n-ensemble", type=int, default=None, help="Regression only: context-resampling runs (default: per dataset).")
    parser.add_argument("--test-chunk-size", type=int, default=None,
                        help="Regression only: score the query set in batches of this size (default: all at once).")
    parser.add_argument("--seed", type=int, default=0, help="Regression only: seed for context resampling (default: %(default)s).")
    parser.add_argument("--output-json", type=Path, default=None,
                        help=f"Metrics JSON (default: {DEFAULT_OUTPUT_ROOT}/<embedding-model>/tabpfnv3_<dataset>.json).")
    args = parser.parse_args()
    if args.dataset in REG_DATASETS:
        for key, value in REG_DATASETS[args.dataset][1].items():
            if getattr(args, key) is None:
                setattr(args, key, value)
    return args


def make_estimator(estimator_cls, checkpoint: str, args: argparse.Namespace):
    """TabPFN v3 with its version defaults, on <--checkpoint-dir>/<checkpoint> (downloaded there by tabpfn if missing)."""
    return estimator_cls.create_default_for_version(
        ModelVersion.V3, n_estimators=args.n_estimators, device=args.device,
        random_state=args.tabpfn_seed, ignore_pretraining_limits=True,
        model_path=str(args.checkpoint_dir / checkpoint),
    )


# --------------------------------------------------------------------------------------------------
# data: embeddings + labels, without torch_geometric
# --------------------------------------------------------------------------------------------------

def load_meta(args: argparse.Namespace) -> tuple[dict, Path]:
    emb_dir = args.embeddings_root / args.embedding_model / args.dataset
    return json.loads((emb_dir / f"{args.dataset}_meta.json").read_text()), emb_dir


def load_embeddings(emb_dir: Path, dataset: str, split: str, n_rows: int) -> np.ndarray:
    emb = np.load(emb_dir / f"{dataset}_embeddings_{split}.npy").astype(np.float32)
    if emb.shape[0] != n_rows:
        raise RuntimeError(f"{emb_dir}: {split} has {emb.shape[0]} embedding rows but {n_rows} meta indices")
    return emb


def moleculenet_labels(datasets_root: Path, dataset: str) -> np.ndarray:
    """(n_csv_rows, n_labels) labels per raw CSV row, exactly as torch_geometric's MoleculeNet builds
    data.y (same line parser, float32, NaN for empty). The CSV must match its split's split_info.json."""
    csv_name, label_col = MOLECULENET[dataset]
    csv_path = datasets_root / "moleculenet" / dataset / "raw" / csv_name
    info = json.loads((datasets_root / "moleculenet" / dataset / "split" / "split_info.json").read_text())
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    if digest != info["source_csv_sha256"]:
        raise RuntimeError(f"{csv_path} is not the CSV the {dataset} scaffold split / embeddings were built from "
                           f"(sha256 {digest[:12]}... != {info['source_csv_sha256'][:12]}...).")
    with open(csv_path) as f:
        lines = [x for x in f.read().split("\n")[1:-1] if len(x) > 0]
    rows = []
    for line in lines:
        labels = re.sub(r"\".*\"", "", line).split(",")[label_col]
        labels = labels if isinstance(labels, list) else [labels]
        rows.append([float(y) if len(y) > 0 else float("nan") for y in labels])
    return np.asarray(rows, dtype=np.float32).astype(np.float64)


class _DictionaryStub:
    """Unpickle stand-in for '__main__.Dictionary', referenced by ZINC's raw pickles."""


def graph_split_labels(datasets_root: Path, dataset: str, split: str) -> np.ndarray:
    """Labels of torch_geometric's ZINC(subset=True, split) / AQSOL(split), in its processing order."""
    raw = datasets_root / dataset / "raw"
    if dataset == "zinc":
        import __main__

        __main__.Dictionary = _DictionaryStub
        with open(raw / f"{PYG_SPLIT[split]}.pickle", "rb") as f:
            mols = pickle.load(f)
        with open(raw / f"{PYG_SPLIT[split]}.index") as f:
            indices = [int(x) for x in f.read()[:-1].split(",")]
        ys = [float(torch.as_tensor(mols[i]["logP_SA_cycle_normalized"]).float()) for i in indices]
    else:
        with open(raw / f"{PYG_SPLIT[split]}.pickle", "rb") as f:
            graphs = pickle.load(f)
        ys = [float(np.float32(y)) for _, _, edge_index, y in graphs if np.asarray(edge_index).size > 0]
    return np.asarray(ys, dtype=np.float64)


def load_split_data(args: argparse.Namespace) -> tuple[dict, dict, Path]:
    """({train|valid|test: (X, labels (n, n_labels))}, meta, embeddings_dir) -- the same molecules and
    labels common.load_all_splits / load_local_index_split give the other evaluation scripts."""
    meta, emb_dir = load_meta(args)
    data = {}
    if args.dataset in MOLECULENET:
        labels = moleculenet_labels(args.datasets_root, args.dataset)
        for split in SPLITS:
            rows = meta["split_idx"][split]
            data[split] = (load_embeddings(emb_dir, args.dataset, split, len(rows)), labels[rows])
    else:
        key = "split_idx" if "split_idx" in meta else "graph_index"
        for split in ("train", "test"):
            rows = [int(i) for i in meta[key][split]]
            labels = graph_split_labels(args.datasets_root, args.dataset, split)
            data[split] = (load_embeddings(emb_dir, args.dataset, split, len(rows)), labels[rows][:, None])
    return data, meta, emb_dir


# --------------------------------------------------------------------------------------------------
# metrics (same definitions as ../common.py)
# --------------------------------------------------------------------------------------------------

def clf_metrics(pos_prob, y_true) -> dict[str, float]:
    """ROC-AUC / AP / accuracy (threshold 0.5); NaN ROC-AUC when a slice is single-class."""
    pos_prob = np.asarray(pos_prob, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    roc_auc = float(sklearn.metrics.roc_auc_score(y_true, pos_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    ap = float(sklearn.metrics.average_precision_score(y_true, pos_prob))
    return {"roc_auc": roc_auc, "ap": ap, "accuracy": float(((pos_prob > 0.5).astype(int) == y_true).mean())}


def three_way_metrics(pos_prob: np.ndarray, target: np.ndarray, n_valid: int) -> dict[str, dict[str, float]]:
    """Query order is [valid..., test...]."""
    return {
        "valid": clf_metrics(pos_prob[:n_valid], target[:n_valid]),
        "test": clf_metrics(pos_prob[n_valid:], target[n_valid:]),
        "combined": clf_metrics(pos_prob, target),
    }


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(target)) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(target))))


REGRESSION_METRICS = {"rmse": rmse, "mae": mae}


def summarize_regression(pred_runs: np.ndarray, target: np.ndarray, metric: str) -> dict:
    """Ensemble-averaged and per-run `metric` (rmse | mae) + R2, in real target units."""
    fn = REGRESSION_METRICS[metric]
    err_runs = np.array([fn(p, target) for p in pred_runs])
    r2_runs = np.array([sklearn.metrics.r2_score(target, p) for p in pred_runs])
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
        "n_ensemble": len(pred_runs),
    }


def avg_over_tasks(per_task: dict, names: list[str], with_std: bool) -> dict:
    """{split -> {metric -> mean}} (ClinTox) or {split -> {metric -> {mean, std_across_tasks}}} (SIDER)."""
    out: dict = {}
    for split in ("valid", "test", "combined"):
        out[split] = {}
        for m in METRIC_KEYS:
            vals = np.array([per_task[n][split][m] for n in names], dtype=np.float64)
            if with_std:
                out[split][m] = {"mean": float(vals.mean()),
                                 "std_across_tasks": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
            else:
                out[split][m] = float(vals.mean())
    return out


def print_cls_block(title: str, per_task: dict, names: list[str], average: dict | None) -> None:
    print(f"\n-- {title} --")
    print(f"{'split':10s} {'task':16s} {'roc_auc':>10s} {'ap':>10s} {'accuracy':>10s}")
    for split in ("valid", "test", "combined"):
        for name in names:
            m = per_task[name][split]
            print(f"{split:10s} {name:16s} {m['roc_auc']:10.4f} {m['ap']:10.4f} {m['accuracy']:10.4f}")
        if average is not None:
            a = average[split]
            vals = [a[k]["mean"] if isinstance(a[k], dict) else a[k] for k in METRIC_KEYS]
            print(f"{split:10s} {'AVG':16s} {vals[0]:10.4f} {vals[1]:10.4f} {vals[2]:10.4f}")


# --------------------------------------------------------------------------------------------------
# classification / regression
# --------------------------------------------------------------------------------------------------

def run_classification(args: argparse.Namespace) -> dict:
    """One TabPFN fit + predict_proba per task: train = context, valid ++ test = query."""
    if args.dataset == "sider":
        tasks, names = args.tasks, [str(t) for t in args.tasks]
    else:
        names = CLS_DATASETS[args.dataset]
        tasks = list(range(len(names)))

    data, meta, emb_dir = load_split_data(args)
    (X_train, lab_train), (X_valid, lab_valid), (X_test, lab_test) = (data[s] for s in SPLITS)
    n_valid = len(X_valid)
    X_query = np.concatenate([X_valid, X_test], axis=0)
    lab_query = np.concatenate([lab_valid, lab_test], axis=0)
    if np.isnan(lab_train[:, tasks]).any() or np.isnan(lab_query[:, tasks]).any():
        raise ValueError(f"{args.dataset}: missing (NaN) labels in the selected task columns -- not supported here.")
    print(f"  context(train)={len(X_train)}, query valid={n_valid}, query test={len(X_test)}; "
          f"features={X_train.shape[1]}; tasks={names}; tabpfn_seed={args.tabpfn_seed}")

    per_task = {}
    for t, name in zip(tasks, names):
        clf = make_estimator(TabPFNClassifier, CLS_CHECKPOINT, args)
        clf.fit(X_train, lab_train[:, t].astype(np.int64))
        proba = np.asarray(clf.predict_proba(X_query), dtype=np.float64)
        pos = proba[:, list(clf.classes_).index(1)]
        per_task[name] = three_way_metrics(pos, lab_query[:, t], n_valid)
        print(f"  task {name}: test roc_auc={per_task[name]['test']['roc_auc']:.4f}")

    average = None
    if args.dataset == "clintox":
        average = avg_over_tasks(per_task, names, with_std=False)
        tabpfn = {"tasks": per_task, "average": average}
    elif args.dataset == "sider":
        # delta-AUPRC = AP - positive rate of the same query slice.
        slices = {"valid": lab_valid, "test": lab_test, "combined": lab_query}
        for t, name in zip(tasks, names):
            for split, lab in slices.items():
                per_task[name][split]["delta_auprc"] = per_task[name][split]["ap"] - float(lab[:, t].mean())
        average = avg_over_tasks(per_task, names, with_std=True)
        tabpfn = {"tasks": per_task, "average": average}
    else:
        tabpfn = per_task[names[0]]

    print_cls_block(f"{args.embedding_model} embeddings -> TabPFN-v3 classification ICL (no pooler, no graphs)",
                    per_task, names, average)
    return {
        "targets": names,
        "embeddings_dir": str(emb_dir),
        "embedding_meta": {k: meta.get(k) for k in ("model", "encoder", "checkpoint", "pooling", "split")},
        "context": "scaffold_split_train_full", "query": "scaffold_split_valid_plus_test",
        "n_context": len(X_train), "n_valid": n_valid, "n_test": len(X_test),
        "tabpfn": tabpfn,
    }


def run_regression(args: argparse.Namespace) -> dict:
    """--n-ensemble TabPFN runs, each on a fresh context subsample, predictions averaged."""
    metric = REG_DATASETS[args.dataset][0]
    err = REGRESSION_METRICS[metric]
    data, meta, emb_dir = load_split_data(args)
    (X_train_full, lab_train), (X_test, lab_test) = data["train"], data["test"]
    y_train_full, y_test = lab_train[:, 0], lab_test[:, 0]
    n_context = min(args.max_train or len(X_train_full), len(X_train_full))
    test_chunk_size = args.test_chunk_size or len(X_test)
    print(f"  train available={len(X_train_full)}, test={len(X_test)}, features={X_train_full.shape[1]}; "
          f"context={n_context}, n_ensemble={args.n_ensemble}, seed={args.seed}, tabpfn_seed={args.tabpfn_seed}")

    pred_runs = []
    for run_idx in range(args.n_ensemble):
        if len(X_train_full) > n_context:
            X_train, _, y_train, _ = train_test_split(
                X_train_full, y_train_full, train_size=n_context, random_state=args.seed + run_idx,
            )
        else:
            X_train, y_train = X_train_full, y_train_full
        reg = make_estimator(TabPFNRegressor, REG_CHECKPOINT, args)
        reg.fit(X_train, y_train)
        chunks = [
            np.asarray(reg.predict(X_test[start:start + test_chunk_size]), dtype=np.float64).reshape(-1)
            for start in range(0, len(X_test), test_chunk_size)
        ]
        pred_runs.append(np.concatenate(chunks))
        print(f"  run {run_idx + 1}/{args.n_ensemble} done ({metric.upper()}={err(pred_runs[-1], y_test):.6f})")

    tabpfn = summarize_regression(np.stack(pred_runs), y_test, metric)
    print(f"\n-- {args.embedding_model} embeddings -> TabPFN-v3 regression ICL: "
          f"ensemble {metric.upper()}={tabpfn[f'ensemble_{metric}']:.6f}, R2={tabpfn['ensemble_r2']:.6f}")
    return {
        "metric": metric,
        "embeddings_dir": str(emb_dir),
        "embedding_meta": {k: meta.get(k) for k in ("model", "encoder", "checkpoint", "pooling", "split")},
        "max_train": args.max_train, "n_ensemble": args.n_ensemble, "seed": args.seed,
        "test_chunk_size": test_chunk_size,
        "n_train_context_used": int(n_context), "n_train_available": len(X_train_full), "n_test": len(X_test),
        "tabpfn": tabpfn,
    }


def main() -> None:
    args = parse_args()
    print(f"=== TabPFN-v3 on {args.dataset} ({args.embedding_model} embeddings) ===")
    run = run_classification(args) if args.dataset in CLS_DATASETS else run_regression(args)

    output_json = args.output_json or (DEFAULT_OUTPUT_ROOT / args.embedding_model / f"tabpfnv3_{args.dataset}.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump({
            "dataset": args.dataset, "model": "TabPFN-v3", "embedding_model": args.embedding_model,
            "tabpfn_seed": args.tabpfn_seed, "n_estimators": args.n_estimators,
            "checkpoint": str(args.checkpoint_dir),
            "checkpoint_version": CLS_CHECKPOINT if args.dataset in CLS_DATASETS else REG_CHECKPOINT,
            **run,
        }, f, indent=2)
    print(f"\nSaved to {output_json}")


if __name__ == "__main__":
    main()
