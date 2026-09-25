"""FS-Mol few-shot eval of a pooler checkpoint using FS-Mol's OWN 32-column atom features for the GNN
AND the per-molecule ECFP fingerprint spliced in as an extra LimiX feature group.

Atom features / graph: exactly as in eval_graphpfn_pooler_on_fsmol_test_native_features.py (stored
node_features + adjacency_lists, bidirectional edges, isolated-atom self-loops). The spliced vector is
the ECFP count fingerprint stored in each task file (2048 dims, optionally folded to --fp-fold),
--per-mol-features descriptors uses the 200-dim RDKit descriptors instead, `both` concatenates
them. It is run through the frozen LimiX feature encoder (one row per MOLECULE, padded to a multiple of
features_per_group, context-normalised with eval_pos = support size) and concatenated onto `pooled`
right after the pooler's output projection.

Two variants are scored in the same forward loop: native_only (atom features alone) and
native_plus_<fp|desc|fpdesc> (with the splice). --no-baseline skips the first.
Usage:
    python eval_graphpfn_pooler_on_fsmol_test_native_plus_fingerprint.py \\
        [--paper-dir /path/to/paper] [--fsmol-dir /path/to/datasets/fs-mol] \\
        [--pooler-checkpoint PATH] [--no-ema] [--device cuda] \\
        [--per-mol-features fingerprint|descriptors|both] [--fp-fold 512] [--no-baseline] \\
        [--support-sizes 16,32,64,128,256] [--num-runs 10] [--seed 0] \\
        [--keep-edge-direction] [--shard-index I --num-shards N] \\
        [--limit-tasks N] [--max-query N] [--output-json PATH]
Slurm job array: submit_fsmol_eval_native_plus_fingerprint_array.sbatch
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))            # fs_mol/ -> the fsmol_core package

from fsmol_core import evaluate, recipes  # noqa: E402

if __name__ == "__main__":
    evaluate.main(recipes.NATIVE_PLUS_FP, __doc__ + "\n" + evaluate.__doc__, __file__, output_dir=SCRIPT_DIR)
