"""SIDER counterpart of eval_clintox.py: same five
pipelines and train-context + valid++test-query convention, over SIDER's 27 side-effect task columns
(labels from PyG's own data.y).

--tasks selects which columns to evaluate and average over: a range like "21-26" (default -- the
PAR / Meta-MGNN "last 6 of 27" meta-test columns), a comma list, or "all".

Metrics per task and averaged (mean +/- std ACROSS the selected tasks), for valid / test / valid+test
combined: ROC-AUC, AP, accuracy, plus delta-AUPRC = AP - positive rate for the pooler pipelines.

Usage:
    python eval_sider.py \\
        --moleculenet-root DIR --pooler-checkpoint PATH --embedding-model {Molbert,MolDeBERTa} \\
        [--tasks 21-26|all] [--embeddings-root DIR] [--no-ema] [--device cpu] \\
        [--skip-embeddings] [--skip-embedding-limix] [--output-json PATH]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import common as ec

DATASET = "sider"
OUTPUT_NAME = "eval_sider"
N_SIDER_TASKS = 27
DEFAULT_TEST_TASKS = "21-26"  # PAR / Meta-MGNN "last 6 of 27" meta-test columns.
_METRIC_KEYS = ("roc_auc", "ap", "accuracy")
_POOLER_KEYS = [k for k in ec.PIPELINE_KEYS if k != "molebert_limix"]


def _parse_tasks(spec: str) -> list[int]:
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
    ec.add_common_args(parser)
    parser.add_argument("--tasks", type=_parse_tasks, default=DEFAULT_TEST_TASKS,
                        help=f"SIDER task columns: 'all', a range like '21-26', or a comma list (default {DEFAULT_TEST_TASKS}).")
    return parser.parse_args()


def _avg_over_tasks(per_task: dict[str, dict], names: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    """{split -> {metric -> {mean, std_across_tasks (ddof=1)}}}."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for split in ("valid", "test", "combined"):
        out[split] = {}
        for m in _METRIC_KEYS:
            vals = np.array([per_task[n][split][m] for n in names], dtype=np.float64)
            out[split][m] = {
                "mean": float(vals.mean()),
                "std_across_tasks": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            }
    return out


def main() -> None:
    args = parse_args()
    tasks: list[int] = args.tasks
    names = [str(t) for t in tasks]
    run = ec.run_classification(args, DATASET, tasks=tasks, task_names=names)
    results = run["results"]

    # delta-AUPRC = AP - positive rate of the same query slice.
    n_context, n_valid = run["n_context"], run["n_valid"]
    labels_all = run["labels_all"]
    slices = {
        "valid": labels_all[n_context:n_context + n_valid],
        "test": labels_all[n_context + n_valid:],
        "combined": labels_all[n_context:],
    }
    for key in _POOLER_KEYS:
        if results[key] is None:
            continue
        for t, n in zip(tasks, names):
            for split, lab in slices.items():
                results[key][n][split]["delta_auprc"] = results[key][n][split]["ap"] - float(lab[:, t].mean())

    metrics = {
        k: ({"tasks": v, "average": _avg_over_tasks(v, names)} if v is not None else None)
        for k, v in results.items()
    }

    print(f"\n=== pooler checkpoint -> real SIDER ({'raw' if args.no_ema else 'EMA'} weights, "
          f"{args.embedding_model} embeddings, self-loops on isolated atoms, {len(tasks)} task(s)) ===")
    for key in ec.PIPELINE_KEYS:
        block = metrics[key]
        ec.print_cls_block(ec.PIPELINE_TITLES[key], results[key], names, block["average"] if block else None)

    output_json = ec.resolve_output_json(args, OUTPUT_NAME)
    with open(output_json, "w") as f:
        json.dump({"dataset": "SIDER", "task_indices": tasks, **ec.common_output_fields(args, run), **metrics}, f, indent=2)
    print(f"\nSaved to {output_json}")


if __name__ == "__main__":
    main()
