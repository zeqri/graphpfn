"""FS-Mol few-shot eval of a pooler checkpoint using FS-Mol's OWN 32-column atom features.

Atom features and graph come straight from each <task>.jsonl.gz (`node_features` + 3 typed
`adjacency_lists`) -- NOT re-parsed from SMILES: no RDKit, nothing is dropped.
  * atom_features  = node_features as-is (32 cols; zero-padded up to a multiple of features_per_group).
  * edges          = union of the 3 typed adjacency lists, made BIDIRECTIONAL (--keep-edge-direction to
                     skip); bond TYPE dropped (the pooler has no edge features).
  * isolated atoms = every zero-in-degree atom gets a src==dst self-loop.
The pooler, the frozen LimiX-16M backbone and the in-context classification forward come from
../../dev_prior_final/ through ../fsmol_core/pooler_glue.py.
Usage:
    python eval_graphpfn_pooler_on_fsmol_test_native_features.py \\
        [--paper-dir /path/to/paper] [--fsmol-dir /path/to/datasets/fs-mol] \\
        [--pooler-checkpoint PATH] [--no-ema] [--device cuda] \\
        [--support-sizes 16,32,64,128,256] [--num-runs 10] [--seed 0] \\
        [--keep-edge-direction] [--shard-index I --num-shards N] \\
        [--limit-tasks N] [--max-query N] [--output-json PATH]
Slurm job array: submit_fsmol_eval_native_features_array.sbatch
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))            # fs_mol/ -> the fsmol_core package

from fsmol_core import evaluate, recipes  # noqa: E402

if __name__ == "__main__":
    evaluate.main(recipes.NATIVE_FEATURES, __doc__ + "\n" + evaluate.__doc__, __file__, output_dir=SCRIPT_DIR)
