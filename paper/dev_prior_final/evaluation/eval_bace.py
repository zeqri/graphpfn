"""Pooler checkpoint -> real BACE (scaffold split: train = context, valid ++ test = query). EVAL-ONLY.

FIVE PIPELINES, one process, same molecules + labels (see common.run_classification):
  1. baseline_extra_features_only            -- atom features = atom_features's 31 RDKit columns alone.
  2. extra_features_only_molebert_augmented  -- 1 + the pretrained molecule embedding (--embedding-model)
                                                spliced into the pooler as one more LimiX feature group.
  3. baseline_x_plus_extra_features           -- atom features = [data.x (9) | 31 RDKit columns].
  4. x_plus_extra_features_molebert_augmented -- 3 + the same embedding splice.
  5. molebert_limix                           -- the embeddings as plain tabular features -> raw
                                                LimiX classification retrieval ICL (no pooler, no graphs).
(Output keys are named "molebert_*" for every embedding model; the model actually used is recorded under
"embedding_model".)

Pipelines 1-4 average --limix-n-members feature-AND-class-permuted forward passes. Zero-degree atoms
get a self-loop. Metrics for valid / test / valid+test combined: ROC-AUC, AP, accuracy.

Usage:
    python eval_bace.py \\
        --moleculenet-root DIR --pooler-checkpoint PATH --embedding-model {Molbert,MolDeBERTa} \\
        [--embeddings-root DIR] [--no-ema] [--device cpu] [--skip-embeddings] \\
        [--skip-embedding-limix] [--output-json PATH]
"""

from __future__ import annotations

import argparse
import json

import common as ec

DATASET = "bace"
OUTPUT_NAME = "eval_bace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ec.add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = ec.run_classification(args, DATASET, tasks=[0], task_names=["Class"])
    metrics = {k: (v["Class"] if v is not None else None) for k, v in run["results"].items()}

    print(f"\n=== pooler checkpoint -> real BACE ({'raw' if args.no_ema else 'EMA'} weights, "
          f"{args.embedding_model} embeddings, self-loops on isolated atoms) ===")
    for key in ec.PIPELINE_KEYS:
        ec.print_cls_block(ec.PIPELINE_TITLES[key], run["results"][key], ["Class"])

    output_json = ec.resolve_output_json(args, OUTPUT_NAME)
    with open(output_json, "w") as f:
        json.dump({"dataset": "BACE", "target": "Class", **ec.common_output_fields(args, run), **metrics}, f, indent=2)
    print(f"\nSaved to {output_json}")


if __name__ == "__main__":
    main()
