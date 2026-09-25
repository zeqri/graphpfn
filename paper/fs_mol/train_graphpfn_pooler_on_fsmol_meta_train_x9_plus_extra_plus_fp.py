"""Episodic meta-training of the pooler on FS-Mol's real Dtrain assays with 40-column atom features
([9-col MoleculeNet-style | 31-col RDKit extra], built from the SMILES via smiles_repair.robust_mol())
PLUS a per-molecule ECFP fingerprint (RAW/unfolded by default, --fp-fold to narrow it) spliced onto
`pooled` AFTER atom-level pooling and BEFORE the transformer encoder. Training counterpart of
evaluation/eval_graphpfn_pooler_on_fsmol_test_x9_features.py (pass --fp-fold there too).
Usage (single GPU):
    python train_graphpfn_pooler_on_fsmol_meta_train_x9_plus_extra_plus_fp.py \\
        [--paper-dir /path/to/paper] [--fsmol-dir /path/to/datasets/fs-mol] \\
        [--pooler-checkpoint PATH] [--device cuda] \\
        [--lr 1e-4] [--n-steps 20000] [--grad-accum-steps 20] \\
        [--support-sizes 16,32,64,128] [--max-query 128] [--fp-fold 0] \\
        [--eval-every 200] [--valid-support-sizes 16,128] [--valid-num-runs 5] [--patience 25] \\
        [--seed 0] [--output-dir DIR]

Usage (4 GPUs, DDP -- same flags, same --output-dir semantics):
    python -m torch.distributed.run --standalone --nproc_per_node=4 train_graphpfn_pooler_on_fsmol_meta_train_x9_plus_extra_plus_fp.py [...]
    (Slurm: submit_fsmol_meta_train_x9_plus_extra_plus_fp_ddp4.sbatch)

Smoke-test first with something small and fast, e.g.:
    python train_graphpfn_pooler_on_fsmol_meta_train_x9_plus_extra_plus_fp.py --n-steps 20 --grad-accum-steps 2 \\
        --eval-every 10 --valid-num-runs 1 --output-dir /tmp/fsmol_meta_train_smoketest
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent            # .../paper/fs_mol/
sys.path.insert(0, str(SCRIPT_DIR))                      # the fsmol_core package

from fsmol_core import recipes, train  # noqa: E402

if __name__ == "__main__":
    train.main(recipes.X9_PLUS_EXTRA_PLUS_FP, __doc__ + "\n" + train.__doc__, output_dir=SCRIPT_DIR / "output" / "fsmol_meta_train_x9_plus_extra_plus_fp")
