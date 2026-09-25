"""FS-Mol few-shot metrics, across-task aggregation, EC-class breakdown and the console report.

Metric: delta-AUPRC = average_precision_score(y_query, P_active) - query positive rate, plus ROC-AUC
(0.0 on a single-class query) and raw AP. Aggregation follows fs_mol.utils.test_utils.eval_model:
mean over runs within a task, then over tasks; error = SEM across tasks (or std across runs for a
single-task EC category). No torch dependency -- the shard-merge scripts import this module too.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np

METRIC_KEYS = ("delta_auprc", "roc_auc", "ap")
EC_LABELS = {
    "1": "oxidoreductases", "2": "kinases (transferases)", "3": "hydrolases", "4": "lyases",
    "5": "isomerases", "6": "ligases", "7": "translocases",
}


def point_metrics(pos_prob, y_true) -> dict:
    """Metrics of ONE (task, support size, run) query set."""
    import sklearn.metrics as skm
    frac_pos = float(y_true.mean())
    ap = float(skm.average_precision_score(y_true, pos_prob))
    roc_auc = float(skm.roc_auc_score(y_true, pos_prob)) if len(np.unique(y_true)) > 1 else 0.0
    return {"delta_auprc": ap - frac_pos, "ap": ap, "roc_auc": roc_auc,
            "frac_pos_query": frac_pos, "n_query": int(len(y_true))}


def agg_across(values) -> dict:
    v = np.array([x for x in values if x is not None and math.isfinite(x)], dtype=np.float64)
    n = len(v)
    std = float(v.std(ddof=0)) if n else float("nan")   # matches metrics.avg_metrics_over_tasks
    return {"mean": float(v.mean()) if n else float("nan"), "std": std,
            "sem": std / math.sqrt(n) if n else float("nan"), "n": n}


def load_ec_map(fsmol_dir: Path, task_names) -> dict:
    """{chembl_id: EC super-class} from <fsmol_dir>/target_info.csv."""
    csv_path = fsmol_dir / "target_info.csv"
    if not csv_path.exists():
        print(f"  [warn] {csv_path} not found -- no EC breakdown")
        return {}
    want = set(task_names)
    out: dict[str, str] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            cid = row.get("chembl_id")
            if cid in want and cid not in out:
                out[cid] = (row.get("EC_super_class") or "").strip() or "unknown"
    return out


def print_ec_counts(ec_by_task: dict) -> None:
    if ec_by_task:
        print(f"  EC_super_class counts: {dict(Counter(ec_by_task.values()))}")


def summarise(per_task: dict, support_sizes: list[int]) -> dict:
    """{size -> {n_tasks, all_enzymes:{metric->agg}, by_ec:{ec->{label,n_tasks,metric->agg}}}} from
    per_task[size][task] = {metric: value, "ec": ..., ["runs": [per-run metric dicts]]}.

    An EC class with a single task reports the std across that task's own runs when `runs` is present
    (live evaluation). Merged shard files do not persist per-run values, so there it falls back to the
    across-task aggregate with n=1 (std = sem = 0)."""
    summary: dict[str, dict] = {}
    for size in support_sizes:
        tasks = per_task[size]
        overall = {k: agg_across([v[k] for v in tasks.values()]) for k in METRIC_KEYS}

        groups: dict[str, list[dict]] = {}
        for v in tasks.values():
            groups.setdefault(v.get("ec", "unknown"), []).append(v)
        by_ec: dict[str, dict] = {}
        for ec, vs in sorted(groups.items()):
            block = {"label": EC_LABELS.get(ec, ec), "n_tasks": len(vs)}
            for k in METRIC_KEYS:
                if len(vs) > 1 or "runs" not in vs[0]:
                    block[k] = agg_across([v[k] for v in vs])
                else:
                    rv = np.array([r[k] for r in vs[0]["runs"]], dtype=np.float64)
                    block[k] = {"mean": float(rv.mean()), "std": float(rv.std(ddof=0)),
                                "sem": float(rv.std(ddof=0) / math.sqrt(len(rv))), "n": len(rv)}
            by_ec[ec] = block

        summary[str(size)] = {"n_tasks": len(tasks), "all_enzymes": overall, "by_ec": by_ec}
    return summary


def print_report(variant: str, summary: dict, support_sizes: list[int], use_ema: bool, atoms_desc: str) -> None:
    print(f"\n=== pooler -> FS-Mol test [{variant}] "
          f"({'EMA' if use_ema else 'raw'} weights, {atoms_desc}) ===")
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

    tbl_size = table_size(support_sizes)
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


def table_size(support_sizes: list[int]) -> int:
    """Support size shown in the Table-2-style breakdown (16 if evaluated, else the smallest)."""
    return 16 if 16 in support_sizes else support_sizes[0]


def print_ab(summaries: dict, variants: list[str], support_sizes: list[int]) -> None:
    """A/B line (baseline variant -> augmented variant) when both were scored."""
    if len(variants) != 2:
        return
    size = table_size(support_sizes)
    base, aug = variants
    b = summaries[base][str(size)]["all_enzymes"]["delta_auprc"]["mean"]
    a = summaries[aug][str(size)]["all_enzymes"]["delta_auprc"]["mean"]
    print(f"\n[A/B @ support {size}]  {base} dAUPRC {b:+.4f}  ->  {aug} dAUPRC {a:+.4f}   (delta {a - b:+.4f})")
