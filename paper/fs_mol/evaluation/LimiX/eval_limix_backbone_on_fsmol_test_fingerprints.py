"""FS-Mol few-shot eval of the RAW LimiX-16M backbone doing in-context classification directly on
FS-Mol's per-molecule ECFP fingerprints -- NO pooler, NO GNN, NO bond graph, NO RealGraphAdapter.

The per-molecule vector is fed to `vendor.limix.inference.predictor.LimiXPredictor` (backbone checkpoint
+ the 16M classification retrieval config) as plain tabular features -- support rows = context, query rows =
predicted.
Here that vector is the 2048-dim ECFP count fingerprint stored in each <task>.jsonl.gz
(`fingerprints`) -- the exact feature the FS-Mol Random-Forest baseline uses, and the same one the
`native_plus_fingerprint` pooler splice uses -- optionally folded (default 2048 -> 512).
`--per-mol-features descriptors` uses the 200-dim RDKit phys-chem `descriptors` field instead;
`both` concatenates folded-fp ++ descriptors.

So this run isolates "what does the frozen LimiX ICL alone get from the fingerprint", with no graph
information at all -- a direct counterpart to RF (a classifier on the same fingerprint) and a floor
for the pooler + fingerprint-splice runs.

Protocol matches the FS-Mol paper (fs_mol/utils/test_utils.py::eval_model, Sec 5.1), identical to the
pooler eval scripts in this directory:
  * 157 held-out test tasks from datasets/fsmol-0.1.json.
  * Support sizes 16/32/64/128/256, --num-runs (10) stratified resamples each, run r seeded seed+r.
  * Split = byte-for-byte reimplementation of StratifiedTaskSampler.sample(train_size=k, valid_size=0,
    test_size=None, allow_smaller_test=True); runs whose split raises are skipped, like eval_model.
  * Per run: clf.predict(X_support, y_support, X_query, task_type="Classification"); P_active = proba[:, 1].
  * Metric: delta-AUPRC = average_precision_score(y_query, P_active) - query positive rate, plus
    ROC-AUC (0.0 on a single-class query) and raw AP. Aggregation: mean over runs within a task,
    then over tasks; error = SEM across tasks (or std across runs for a single-task EC category).
    Broken out by EC super-class (target_info.csv column EC_super_class).

Needs `vendor.limix` (in <paper>/vendor/limix) and the backbone checkpoint <paper>/checkpoints/LimiX-16M.ckpt; the
retrieval inference config effectively requires CUDA. No pooler checkpoint, no rdkit, no torch_geometric, no
bond graphs.

SHARDING: --shard-index/--num-shards split the task list round-robin (`tasks[shard_index::num_shards]`);
each shard writes its own '.shard{I}of{N}.json'. Leave them at their defaults (one shard) to evaluate all
157 tasks in a single process and get a complete, paper-comparable result.

Usage:
    python eval_limix_backbone_on_fsmol_test_fingerprints.py \\
        [--paper-dir DIR] [--fsmol-dir DIR] \\
        [--backbone-checkpoint PATH] [--inference-config PATH] [--device cuda] \\
        [--per-mol-features fingerprint|descriptors|both] [--fp-fold 512] \\
        [--support-sizes 16,32,64,128,256] [--num-runs 10] [--seed 0] \\
        [--shard-index I --num-shards N] \\
        [--limit-tasks N] [--max-query N] [--output-json PATH]
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent             # <paper>/fs_mol/evaluation/LimiX/
PAPER_DIR = SCRIPT_DIR.parents[2]                        # <paper>/
FSMOL_DIR_DEFAULT = PAPER_DIR.parent / "datasets" / "fs-mol"
BACKBONE_CHECKPOINT_DEFAULT = PAPER_DIR / "checkpoints" / "LimiX-16M.ckpt"
INFERENCE_CONFIG_DEFAULT = PAPER_DIR / "vendor" / "limix" / "config" / "cls_default_16M_retrieval.json"
OUTPUT_DIR_DEFAULT = SCRIPT_DIR / "output"

DEFAULT_SUPPORT_SIZES = "16,32,64,128,256"
_METRIC_KEYS = ("delta_auprc", "roc_auc", "ap")
EC_LABELS = {
    "1": "oxidoreductases", "2": "kinases (transferases)", "3": "hydrolases", "4": "lyases",
    "5": "isomerases", "6": "ligases", "7": "translocases",
}
FP_LEN = 2048     # ECFP count-fingerprint width stored in FS-Mol .jsonl.gz
DESC_LEN = 200    # RDKit phys-chem descriptor width stored in FS-Mol .jsonl.gz

def _bootstrap_paper(paper_dir: Path) -> None:
    """Put <paper> on sys.path so `vendor.limix` imports."""
    if not (paper_dir / "vendor" / "limix").exists():
        raise SystemExit(f"--paper-dir {paper_dir} does not contain vendor/limix/.")
    if str(paper_dir) not in sys.path:
        sys.path.insert(0, str(paper_dir))


# --------------------------------------------------------------------------------------------------
# args
# --------------------------------------------------------------------------------------------------

def _parse_sizes(spec: str) -> list[int]:
    out = sorted({int(p) for p in spec.replace(" ", "").split(",") if p})
    if not out:
        raise argparse.ArgumentTypeError("empty --support-sizes")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paper-dir", type=Path, default=PAPER_DIR,
                   help="'paper' dir holding vendor/limix/ and checkpoints/. Default: the paper/ dir above this script.")
    p.add_argument("--fsmol-dir", type=Path, default=FSMOL_DIR_DEFAULT,
                   help="FS-Mol data dir (test/, fsmol-0.1.json, target_info.csv). Default: <repo>/datasets/fs-mol.")
    p.add_argument("--fsmol-data", type=Path, default=None, help="Dir with test/ *.jsonl.gz (default: <fsmol-dir>).")
    p.add_argument("--task-list-json", type=Path, default=None,
                   help="JSON with a 'test' list (default: <fsmol-dir>/fsmol-0.1.json).")
    p.add_argument("--backbone-checkpoint", type=Path, default=None,
                   help="LimiX backbone checkpoint (default: <paper-dir>/checkpoints/LimiX-16M.ckpt).")
    p.add_argument("--inference-config", type=Path, default=None,
                   help="LimiX inference config json (default: <paper-dir>/vendor/limix/config/cls_default_16M_retrieval.json).")
    p.add_argument("--device", default=None, help="Default: cuda if available else cpu (retrieval config effectively needs cuda).")
    p.add_argument("--per-mol-features", choices=("fingerprint", "descriptors", "both"), default="fingerprint",
                   help="Tabular feature source. Default: fingerprint (ECFP, the RF baseline feature).")
    p.add_argument("--fp-fold", type=int, default=512,
                   help="Fold the 2048-dim ECFP to this width (classic modulo folding; <=0 or >=2048 = no folding). "
                        "Default 512. Ignored for --per-mol-features descriptors.")
    p.add_argument("--support-sizes", type=_parse_sizes, default=None,
                   help=f"Comma list of support sizes (default: {DEFAULT_SUPPORT_SIZES}).")
    p.add_argument("--num-runs", type=int, default=10, help="Resamples per (task, support size). FS-Mol default: 10.")
    p.add_argument("--seed", type=int, default=0, help="Base seed; run r uses seed+r. Default: 0.")
    p.add_argument("--tasks", type=str, default=None,
                   help="Comma-separated task names (ChEMBL ids) instead of all 157. NOT paper-comparable.")
    p.add_argument("--limit-tasks", type=int, default=None, help="First N tasks only (NOT paper-comparable).")
    p.add_argument("--max-query", type=int, default=None,
                   help="Cap query set to N mols per run. Default: None = full query (paper).")
    p.add_argument("--shard-index", type=int, default=0,
                   help="This worker's index in [0, num-shards). Default: 0.")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total number of parallel workers; the task list is split round-robin "
                        "(tasks[shard_index::num_shards]) across them. Default: 1 (no sharding). "
                        "Merge shard outputs with ./merge_limix_shards.py.")
    p.add_argument("--output-json", type=Path, default=None,
                   help="Default: <script dir>/output/eval_limix_backbone_on_fsmol_test_fingerprints.json "
                        "(a '.shard{I}of{N}.json' suffix is used instead when --num-shards > 1).")
    args = p.parse_args()
    if not (0 <= args.shard_index < max(args.num_shards, 1)):
        raise SystemExit(f"--shard-index {args.shard_index} must be in [0, --num-shards={args.num_shards})")
    return args


# --------------------------------------------------------------------------------------------------
# FS-Mol data  (only labels + per-molecule fingerprint / descriptors -- graph is not read)
# --------------------------------------------------------------------------------------------------

def read_fsmol_task_permol(path: Path):
    """(labels, fps, descs) for one <task>.jsonl.gz, in file order.
    fps  : ndarray [N, 2048]   descs : ndarray [N, 200]"""
    import numpy as np
    labels, fps, descs = [], [], []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            d = json.loads(line)
            labels.append(int(bool(float(d["Property"]))))
            fp = d.get("fingerprints")
            fps.append(np.zeros(FP_LEN, np.float32) if fp is None else np.asarray(fp, np.float32))
            de = d.get("descriptors")
            descs.append(np.full(DESC_LEN, np.nan, np.float32) if de is None else np.asarray(de, np.float32))
    return np.asarray(labels, np.int64), np.stack(fps, 0), np.stack(descs, 0)


def fold_fingerprint(fp, n_folded: int):
    """Classic ECFP folding: folded[:, j % n_folded] += fp[:, j].  fp: [N, D] -> [N, n_folded].
    n_folded <= 0 or >= D returns fp unchanged. D is zero-padded to a multiple of n_folded first."""
    import numpy as np
    n, d = fp.shape
    if n_folded <= 0 or n_folded >= d:
        return fp.astype(np.float32)
    pad = (-d) % n_folded
    if pad:
        fp = np.concatenate([fp, np.zeros((n, pad), fp.dtype)], axis=1)
    return fp.reshape(n, -1, n_folded).sum(axis=1).astype(np.float32)


def build_permol_features(fps, descs, mode: str, fp_fold: int):
    """[N, F] tabular matrix. fingerprint -> folded ECFP; descriptors -> raw RDKit descriptors
    with +/-inf -> NaN; both -> concat."""
    import numpy as np
    parts = []
    if mode in ("fingerprint", "both"):
        parts.append(fold_fingerprint(fps, fp_fold))
    if mode in ("descriptors", "both"):
        d = np.array(descs, dtype=np.float32)
        d[np.isinf(d)] = np.nan
        parts.append(d)
    return np.concatenate(parts, axis=1)


# --------------------------------------------------------------------------------------------------
# stratified support/query split -- byte-for-byte reimplementation of
# fs_mol.data.fsmol_task_sampler.StratifiedTaskSampler.sample(train_size=k, valid_size=0,
# test_size=None, allow_smaller_test=True)   [copied from the pooler eval scripts]
# --------------------------------------------------------------------------------------------------

class _SplitTooSmall(Exception):
    """Stand-in for FS-Mol's DatasetTooSmall / FoldTooSmall -- caller skips the run."""


def stratified_support_query(labels, support_size: int, seed: int):
    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit

    n = len(labels)
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    order = np.concatenate([pos, neg])                                        # FS-Mol `samples` order
    strat = np.concatenate([np.zeros(len(pos), int), np.ones(len(neg), int)])  # FS-Mol `labels`

    num_test = n - support_size
    if support_size >= n or num_test < 2:
        raise _SplitTooSmall(f"n={n}, support={support_size}, query={num_test}")
    sss = StratifiedShuffleSplit(n_splits=1, train_size=support_size, test_size=num_test, random_state=seed)
    try:
        tr, te = next(iter(sss.split(np.arange(n), strat)))
    except ValueError as e:
        raise _SplitTooSmall(str(e))

    support_idx, query_idx = order[tr], order[te]
    if len(query_idx) < 2:
        raise _SplitTooSmall("query fold < 2")
    for nm, sl in (("support", support_idx), ("query", query_idx)):
        npos = int(labels[sl].sum())
        if not (0 < npos < len(sl)):
            raise _SplitTooSmall(f"{nm} fold single-class ({npos}/{len(sl)})")
    return support_idx, query_idx


# --------------------------------------------------------------------------------------------------
# LimiX tabular in-context classification
# --------------------------------------------------------------------------------------------------

def predict_limix(clf, X_sup, y_sup, X_qry):
    """proba[:, 1] (P active) for X_qry given (X_sup, y_sup) as context -- same call as the BACE
    run_molebert_limix_pipeline."""
    import numpy as np
    import torch
    proba = clf.predict(X_sup, y_sup, X_qry, task_type="Classification")
    if isinstance(proba, torch.Tensor):
        proba = proba.detach().float().cpu().numpy()
    proba = np.asarray(proba, dtype=np.float64)
    return proba[:, 1]


# --------------------------------------------------------------------------------------------------
# metrics / aggregation   [copied from the pooler eval scripts]
# --------------------------------------------------------------------------------------------------

def _point_metrics(pos_prob, y_true):
    import numpy as np
    import sklearn.metrics as skm
    frac_pos = float(y_true.mean())
    ap = float(skm.average_precision_score(y_true, pos_prob))
    roc_auc = float(skm.roc_auc_score(y_true, pos_prob)) if len(np.unique(y_true)) > 1 else 0.0
    return {"delta_auprc": ap - frac_pos, "ap": ap, "roc_auc": roc_auc,
            "frac_pos_query": frac_pos, "n_query": int(len(y_true))}


def _agg_across(values):
    import numpy as np
    v = np.array([x for x in values if x is not None and math.isfinite(x)], dtype=np.float64)
    n = len(v)
    std = float(v.std(ddof=0)) if n else float("nan")
    return {"mean": float(v.mean()) if n else float("nan"), "std": std,
            "sem": std / math.sqrt(n) if n else float("nan"), "n": n}


def load_ec_map(fsmol_dir: Path, task_names):
    csv_path = fsmol_dir / "target_info.csv"
    if not csv_path.exists():
        print(f"  [warn] {csv_path} not found -- no EC breakdown")
        return {}
    import csv
    want = set(task_names)
    out: dict[str, str] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            cid = row.get("chembl_id")
            if cid in want and cid not in out:
                out[cid] = (row.get("EC_super_class") or "").strip() or "unknown"
    return out


def summarise(per_task: dict, support_sizes: list[int]) -> dict:
    import numpy as np
    summary: dict[str, dict] = {}
    for size in support_sizes:
        tasks = per_task[size]
        overall = {k: _agg_across([v[k] for v in tasks.values()]) for k in _METRIC_KEYS}

        by_ec: dict[str, dict] = {}
        ec_groups: dict[str, list[dict]] = {}
        for v in tasks.values():
            ec_groups.setdefault(v["ec"], []).append(v)
        for ec, vs in sorted(ec_groups.items()):
            block = {"label": EC_LABELS.get(ec, ec), "n_tasks": len(vs)}
            for k in _METRIC_KEYS:
                if len(vs) > 1:
                    block[k] = _agg_across([v[k] for v in vs])
                else:
                    rv = np.array([r[k] for r in vs[0]["runs"]], dtype=np.float64)
                    block[k] = {"mean": float(rv.mean()), "std": float(rv.std(ddof=0)),
                                "sem": float(rv.std(ddof=0) / math.sqrt(len(rv))), "n": len(rv)}
            by_ec[ec] = block

        summary[str(size)] = {"n_tasks": len(tasks), "all_enzymes": overall, "by_ec": by_ec}
    return summary


def print_report(summary: dict, support_sizes: list[int], tag: str) -> None:
    print(f"\n=== RAW LimiX-16M backbone -> FS-Mol test [{tag}] (tabular ICL on the per-molecule vector, no GNN/pooler) ===")
    print("Figure-2a-style curve (mean over tasks +/- SEM across tasks):")
    print(f"  {'support':>8s} {'n_tasks':>8s} {'delta-AUPRC':>22s} {'ROC-AUC':>18s} {'AP':>18s}")
    for size in support_sizes:
        a = summary[str(size)]
        if a["n_tasks"] == 0:
            print(f"  {size:8d} {0:8d}   (no evaluable tasks)")
            continue
        d, r, p = a["all_enzymes"]["delta_auprc"], a["all_enzymes"]["roc_auc"], a["all_enzymes"]["ap"]
        print(f"  {size:8d} {a['n_tasks']:8d}   {d['mean']:+.4f} +/- {d['sem']:.4f}"
              f"     {r['mean']:.4f} +/- {r['sem']:.4f}   {p['mean']:.4f} +/- {p['sem']:.4f}")

    tbl_size = 16 if 16 in support_sizes else support_sizes[0]
    print(f"\nTable-2-style breakdown at support size {tbl_size} (delta-AUPRC, mean +/- error):")
    print(f"  {'class':>5s}  {'description':22s} {'#tasks':>7s}  {'delta-AUPRC':>18s}")
    b = summary[str(tbl_size)]["by_ec"]
    for ec in sorted(b, key=lambda k: (not k.isdigit(), int(k) if k.isdigit() else 0)):
        blk = b[ec]
        m = blk["delta_auprc"]
        print(f"  {ec:>5s}  {blk['label']:22s} {blk['n_tasks']:7d}  {m['mean']:+.4f} +/- {m.get('sem', m.get('std')):.4f}")
    allblk = summary[str(tbl_size)]["all_enzymes"]["delta_auprc"]
    print(f"  {'all':>5s}  {'all enzymes':22s} {summary[str(tbl_size)]['n_tasks']:7d}  "
          f"{allblk['mean']:+.4f} +/- {allblk['sem']:.4f}")


# --------------------------------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    _bootstrap_paper(args.paper_dir)

    import numpy as np
    import torch
    from vendor.limix.inference.predictor import LimiXPredictor  # noqa: E402

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        print("  [warn] LimiX retrieval inference config effectively needs CUDA -- CPU may be very slow or fail.")
    support_sizes = args.support_sizes or _parse_sizes(DEFAULT_SUPPORT_SIZES)
    backbone_ckpt = args.backbone_checkpoint or (args.paper_dir / "checkpoints" / "LimiX-16M.ckpt")
    inference_config = args.inference_config or (args.paper_dir / "vendor" / "limix" / "config" / "cls_default_16M_retrieval.json")

    mode_tag = {"fingerprint": "fp", "descriptors": "desc", "both": "fpdesc"}[args.per_mol_features]
    tag = f"limix_{mode_tag}"

    fsmol_dir = args.fsmol_dir
    data_dir = args.fsmol_data or fsmol_dir
    test_dir = data_dir / "test"
    task_list_json = args.task_list_json or (fsmol_dir / "fsmol-0.1.json")

    all_test_task_names = sorted(json.loads(task_list_json.read_text())["test"])
    test_task_names = all_test_task_names
    if args.tasks is not None:
        wanted = [t.strip() for t in args.tasks.split(",") if t.strip()]
        unknown = [t for t in wanted if t not in set(all_test_task_names)]
        if unknown:
            raise SystemExit(f"--tasks: not in {task_list_json} 'test' list: {unknown}")
        test_task_names = [t for t in all_test_task_names if t in set(wanted)]
        print(f"[warn] --tasks: evaluating {len(test_task_names)} task(s) {test_task_names} -- NOT comparable to the paper's 157-task numbers")
    if args.limit_tasks is not None:
        test_task_names = test_task_names[: args.limit_tasks]
        print(f"[warn] --limit-tasks {args.limit_tasks}: NOT comparable to the paper's 157-task numbers")
    if args.max_query is not None:
        print(f"[warn] --max-query {args.max_query}: query set is capped, NOT the paper's full-query protocol")

    sharded = args.num_shards > 1
    if sharded:
        full_count = len(test_task_names)
        test_task_names = test_task_names[args.shard_index :: args.num_shards]
        print(f"[shard {args.shard_index}/{args.num_shards}] {len(test_task_names)}/{full_count} tasks "
              f"(round-robin tasks[{args.shard_index}::{args.num_shards}]) -- "
              f"combine all shards with merge_limix_shards.py for the paper-comparable number")

    print(f"FS-Mol test tasks: {len(test_task_names)} (from {task_list_json})")
    print(f"data dir: {data_dir}  |  tabular feature: {args.per_mol_features}"
          + (f" (ECFP folded {FP_LEN} -> {args.fp_fold})" if args.per_mol_features != "descriptors" and 0 < args.fp_fold < FP_LEN else "")
          + "  |  NO GNN / pooler / bond graph")
    print(f"support sizes: {support_sizes}  |  runs/size: {args.num_runs}")
    print(f"backbone: {backbone_ckpt}")
    print(f"inference config: {inference_config}")

    ec_by_task = load_ec_map(fsmol_dir, test_task_names)
    if ec_by_task:
        from collections import Counter
        print(f"  EC_super_class counts: {dict(Counter(ec_by_task.values()))}")

    # ---- predictor (once) --------------------------------------------------------------------
    # LimiXPredictor(device=<torch.device>, model_path=<backbone ckpt>, inference_config=<cls retrieval json>).
    print("Loading LimiXPredictor (classification retrieval) ...")
    clf = LimiXPredictor(device=device, model_path=str(backbone_ckpt), inference_config=str(inference_config))

    # ---- per-task loop --------------------------------------------------------------------
    per_task: dict[int, dict[str, dict]] = {s: {} for s in support_sizes}
    permol_width = None
    t0 = time.time()

    for ti, name in enumerate(test_task_names):
        labels, fps, descs = read_fsmol_task_permol(test_dir / f"{name}.jsonl.gz")
        X_all = build_permol_features(fps, descs, args.per_mol_features, args.fp_fold)
        permol_width = X_all.shape[1]
        n_pos = int(labels.sum())
        print(f"[{ti + 1:3d}/{len(test_task_names)}] {name}: {len(labels)} mols "
              f"({n_pos} pos / {len(labels) - n_pos} neg)  X[{permol_width}]  [{time.time() - t0:.0f}s]")
        if len(labels) < 4 or n_pos == 0 or n_pos == len(labels):
            print("    skipping (degenerate task)")
            continue

        for size in support_sizes:
            run_points: list[dict] = []
            printed_skip = False
            for run_idx in range(args.num_runs):
                seed = args.seed + run_idx
                try:
                    sup_idx, qry_idx = stratified_support_query(labels, size, seed)
                except _SplitTooSmall as e:
                    if not printed_skip:
                        print(f"    support {size:3d}: run skipped ({e})")
                        printed_skip = True
                    continue

                if args.max_query is not None and len(qry_idx) > args.max_query:
                    rng = np.random.default_rng(seed)
                    qry_idx = np.sort(rng.choice(qry_idx, size=args.max_query, replace=False))

                X_sup, y_sup = X_all[sup_idx], labels[sup_idx]
                X_qry, y_qry = X_all[qry_idx], labels[qry_idx]
                try:
                    pos_prob = predict_limix(clf, X_sup, y_sup, X_qry)
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        print(f"    support {size:3d} run {run_idx}: OOM on {len(qry_idx)} query mols "
                              f"-- skipped (use --max-query / smaller --fp-fold)")
                        continue
                    raise
                run_points.append(_point_metrics(pos_prob, y_qry))

            if run_points:
                per_task[size][name] = {
                    **{k: float(np.mean([p[k] for p in run_points])) for k in
                       ("delta_auprc", "roc_auc", "ap", "frac_pos_query")},
                    "n_runs": len(run_points),
                    "runs": run_points,
                    "ec": ec_by_task.get(name, "unknown"),
                }
                t = per_task[size][name]
                print(f"    support {size:3d}: dAUPRC={t['delta_auprc']:+.4f}  AUROC={t['roc_auc']:.4f}  "
                      f"AP={t['ap']:.4f}  ({t['n_runs']} runs)")

    # ---- aggregate + report -------------------------------------------------------------------
    summary = summarise(per_task, support_sizes)
    print_report(summary, support_sizes, tag)

    # ---- save --------------------------------------------------------------------------
    default_name = "eval_limix_backbone_on_fsmol_test_fingerprints.json"
    if sharded:
        default_name = f"eval_limix_backbone_on_fsmol_test_fingerprints.shard{args.shard_index}of{args.num_shards}.json"
    out = args.output_json or (OUTPUT_DIR_DEFAULT / default_name)
    payload = {
        "dataset": "FS-Mol",
        "method": "raw_limix16m_tabular_icl",
        "tabular_feature": args.per_mol_features,
        "fp_len_raw": FP_LEN,
        "fp_fold": args.fp_fold,
        "feature_width": permol_width,
        "gnn": False, "pooler": False, "bond_graph": False,
        "protocol": "fs_mol.utils.test_utils.eval_model (StratifiedTaskSampler, test_size_or_ratio=None)",
        "task_list_json": str(task_list_json),
        "n_test_tasks": len(test_task_names),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "metric": "delta_auprc = average_precision_score(y_query, P_active) - query_positive_rate",
        "backbone_checkpoint": str(backbone_ckpt),
        "inference_config": str(inference_config),
        "support_sizes": support_sizes,
        "num_runs": args.num_runs,
        "seed": args.seed,
        "max_query": args.max_query,
        "variants": [tag],
        "paper_comparable": (
            args.limit_tasks is None and args.max_query is None and args.tasks is None and not sharded
        ),  # a single shard's own JSON is a PARTIAL result -- merge shards for the paper-comparable number
        "summary": {tag: summary},
        "per_task": {
            tag: {
                str(size): {
                    tn: {k: val[k] for k in ("delta_auprc", "roc_auc", "ap", "frac_pos_query", "n_runs", "ec")}
                    for tn, val in per_task[size].items()
                }
                for size in support_sizes
            }
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
