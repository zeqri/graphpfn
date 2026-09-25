"""Combine the shard JSONs of a sharded eval_graphpfn_pooler_on_fsmol_test_x9_features.py run into one paper-comparable result.
Usage:
    python merge_x9_features_shards.py \\
        --glob "output/eval_graphpfn_pooler_on_fsmol_test_x9_features.shard*of*.json" \\
        --output-json output/eval_graphpfn_pooler_on_fsmol_test_x9_features_merged.json
    # or list the shard files explicitly:
    python merge_x9_features_shards.py shard0.json shard1.json ... --output-json merged.json
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))            # fs_mol/ -> the fsmol_core package

from fsmol_core import merge, recipes  # noqa: E402

if __name__ == "__main__":
    merge.main(eval_stem="eval_graphpfn_pooler_on_fsmol_test_x9_features", output_dir=SCRIPT_DIR / "output", atoms_desc=recipes.X9_PLUS_EXTRA_PLUS_FP.atoms_desc,
               description=merge.__doc__ + "\n" + __doc__)
