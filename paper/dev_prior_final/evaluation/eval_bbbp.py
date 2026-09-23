"""BBBP counterpart of eval_bace.py -- same five
pipelines (see that script / common.run_classification), target p_np.

ALIGNMENT: the embeddings meta's split_idx holds BBBP CSV rows (2050 rows; the 11 RDKit-unparseable
ones are already excluded). PyG's MoleculeNet(name="bbbp") drops those same rows (2039 molecules),
and common.csv_row_to_pyg_index maps every kept CSV row to its PyG index.

Usage:
    python eval_bbbp.py \\
        --moleculenet-root DIR --pooler-checkpoint PATH --embedding-model {Molbert,MolDeBERTa} \\
        [--embeddings-root DIR] [--no-ema] [--device cpu] [--skip-embeddings] \\
        [--skip-embedding-limix] [--output-json PATH]
"""

from __future__ import annotations

import argparse
import json

import common as ec

DATASET = "bbbp"
OUTPUT_NAME = "eval_bbbp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ec.add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = ec.run_classification(args, DATASET, tasks=[0], task_names=["p_np"])
    metrics = {k: (v["p_np"] if v is not None else None) for k, v in run["results"].items()}

    print(f"\n=== pooler checkpoint -> real BBBP ({'raw' if args.no_ema else 'EMA'} weights, "
          f"{args.embedding_model} embeddings, self-loops on isolated atoms) ===")
    for key in ec.PIPELINE_KEYS:
        ec.print_cls_block(ec.PIPELINE_TITLES[key], run["results"][key], ["p_np"])

    output_json = ec.resolve_output_json(args, OUTPUT_NAME)
    with open(output_json, "w") as f:
        json.dump({"dataset": "BBBP", "target": "p_np", **ec.common_output_fields(args, run), **metrics}, f, indent=2)
    print(f"\nSaved to {output_json}")


if __name__ == "__main__":
    main()
