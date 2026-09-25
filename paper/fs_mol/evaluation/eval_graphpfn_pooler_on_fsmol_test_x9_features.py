"""FS-Mol few-shot eval of a pooler checkpoint using the CONCATENATION of two atom-feature blocks
(40 columns) built from each molecule's SMILES, plus the per-molecule ECFP fingerprint splice.

  * the 9-column "MoleculeNet-style" block -- torch_geometric's from_smiles atom features
    (atomic_num, chirality, degree, formal_charge, num_hs, num_radical_electrons, hybridization,
    is_aromatic, is_in_ring);
  * the 31-column RDKit "extra" block (../../dev_prior_final/evaluation/atom_features.py).
Both blocks are read off the SAME RDKit mol, built via smiles_repair.robust_mol(), so the systematic
[PH](=O)(=O)O phosphorus-valence SMILES defect is repaired instead of dropping molecules. A task
whose molecules cannot all be built (unparseable, or atom count != FS-Mol's node_features count) is
skipped as a whole and listed under `task_errors` -- a subset is never silently dropped. The graph
(bonds) still comes from FS-Mol's stored adjacency_lists.

Variants scored in the same loop: x9_plus_extra_only (atom features alone) and
x9_plus_extra_plus_<fp|desc|fpdesc> (with the per-molecule splice). --no-baseline skips the first.
Usage:
    python eval_graphpfn_pooler_on_fsmol_test_x9_features.py \\
        [--paper-dir /path/to/paper] [--fsmol-dir /path/to/datasets/fs-mol] \\
        [--pooler-checkpoint PATH] [--no-ema] [--device cuda] \\
        [--per-mol-features fingerprint|descriptors|both] [--fp-fold 512] [--no-baseline] \\
        [--support-sizes 16,32,64,128,256] [--num-runs 10] [--seed 0] \\
        [--keep-edge-direction] [--shard-index I --num-shards N] \\
        [--limit-tasks N] [--max-query N] [--output-json PATH]
Slurm job array: submit_fsmol_eval_x9_features.sbatch
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))            # fs_mol/ -> the fsmol_core package

from fsmol_core import evaluate, recipes  # noqa: E402

if __name__ == "__main__":
    evaluate.main(recipes.X9_PLUS_EXTRA_PLUS_FP, __doc__ + "\n" + evaluate.__doc__, __file__, output_dir=SCRIPT_DIR / "output")
