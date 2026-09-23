"""ClinTox counterpart of eval_bace.py. ClinTox has
TWO independent binary tasks (FDA_APPROVED, CT_TOX): every pipeline runs once per task and is also
reported averaged over the two.

LABELS come from PyG's own data.y ([1, 2], columns FDA_APPROVED, CT_TOX -- MoleculeNet reads CSV
columns 1:3), NOT from saved label .npy files. ALIGNMENT: the embeddings meta's split_idx holds
ClinTox CSV rows; common.csv_row_to_pyg_index maps them onto PyG's 1479 kept molecules. Rows
that failed to encode for a given embedding model (e.g. Molbert's converter failures) are already
absent from that model's split_idx, so the context set can differ by a few molecules between
--embedding-model choices.

Usage:
    python eval_clintox.py \\
        --moleculenet-root DIR --pooler-checkpoint PATH --embedding-model {Molbert,MolDeBERTa} \\
        [--embeddings-root DIR] [--no-ema] [--device cpu] [--skip-embeddings] \\
        [--skip-embedding-limix] [--output-json PATH]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import common as ec

DATASET = "clintox"
OUTPUT_NAME = "eval_clintox"
TASKS = ["FDA_APPROVED", "CT_TOX"]  # column order of PyG's ClinTox data.y
_METRIC_KEYS = ("roc_auc", "ap", "accuracy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ec.add_common_args(parser)
    return parser.parse_args()


def _avg_over_tasks(per_task: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    return {
        split: {m: float(np.mean([per_task[t][split][m] for t in TASKS])) for m in _METRIC_KEYS}
        for split in ("valid", "test", "combined")
    }


def main() -> None:
    args = parse_args()
    run = ec.run_classification(args, DATASET, tasks=list(range(len(TASKS))), task_names=TASKS)
    metrics = {
        k: ({"tasks": v, "average": _avg_over_tasks(v)} if v is not None else None)
        for k, v in run["results"].items()
    }

    print(f"\n=== pooler checkpoint -> real ClinTox ({'raw' if args.no_ema else 'EMA'} weights, "
          f"{args.embedding_model} embeddings, self-loops on isolated atoms) ===")
    for key in ec.PIPELINE_KEYS:
        block = metrics[key]
        ec.print_cls_block(ec.PIPELINE_TITLES[key], run["results"][key], TASKS, block["average"] if block else None)

    output_json = ec.resolve_output_json(args, OUTPUT_NAME)
    with open(output_json, "w") as f:
        json.dump({"dataset": "ClinTox", "targets": TASKS, **ec.common_output_fields(args, run), **metrics}, f, indent=2)
    print(f"\nSaved to {output_json}")


if __name__ == "__main__":
    main()
