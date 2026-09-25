"""Feature recipes: what goes into the pooler for one FS-Mol trainer / evaluator.

A recipe fixes (a) the ATOM features and (b) whether a per-molecule ECFP fingerprint is spliced onto
the pooled atom representation before the transformer:

    native_features           FS-Mol's own 32-col `node_features` (no RDKit)
    native_plus_fp            native_features + spliced per-molecule fingerprint
    x9_plus_extra_plus_fp     [9-col MoleculeNet-style | 31-col RDKit extra] = 40 cols, built from the
                              SMILES via robust_mol(), + spliced per-molecule fingerprint

Everything else (task IO, split, graph batching, metrics, training loop) is shared -- see data.py,
evaluate.py and train.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from . import data, pooler_glue, smiles_repair

# per-molecule splice mode -> suffix of the augmented variant's name
MODE_TAGS = {"fingerprint": "fp", "descriptors": "desc", "both": "fpdesc"}


class UnbuildableTask(ValueError):
    """The recipe's atom features could not be built for every molecule of a task (task is skipped
    as a whole -- a subset of its molecules is never silently dropped)."""


@dataclass(frozen=True)
class Recipe:
    name: str                       # short id, also names the output dir
    atoms: str                      # "native" | "x9_plus_extra"
    splice: bool                    # per-molecule fingerprint splice on top of the atom features
    atoms_desc: str                 # for console output
    baseline_variant: str           # variant name for "atom features alone"
    augmented_prefix: str = ""      # augmented variant = prefix + MODE_TAGS[mode]
    flat_json: bool = False         # eval JSON keyed by support size directly (no variants level)
    eval_script: str = ""           # file names, for hints in console output
    merge_script: str = ""

    def augmented_variant(self, per_mol_mode: str) -> str:
        return self.augmented_prefix + MODE_TAGS[per_mol_mode]

    def variants(self, per_mol_mode: str, run_baseline: bool) -> list[str]:
        if not self.splice:
            return [self.baseline_variant]
        return ([self.baseline_variant] if run_baseline else []) + [self.augmented_variant(per_mol_mode)]


NATIVE_FEATURES = Recipe(
    name="native_features", atoms="native", splice=False,
    atoms_desc="FS-Mol native 32-col atom features", baseline_variant="native", flat_json=True,
    eval_script="eval_graphpfn_pooler_on_fsmol_test_native_features.py",
    merge_script="merge_native_features_shards.py",
)
NATIVE_PLUS_FP = Recipe(
    name="native_features_plus_fp", atoms="native", splice=True,
    atoms_desc="FS-Mol native 32-col atom features", baseline_variant="native_only",
    augmented_prefix="native_plus_",
    eval_script="eval_graphpfn_pooler_on_fsmol_test_native_plus_fingerprint.py",
    merge_script="merge_native_plus_fingerprint_shards.py",
)
X9_PLUS_EXTRA_PLUS_FP = Recipe(
    name="x9_plus_extra_plus_fp", atoms="x9_plus_extra", splice=True,
    atoms_desc="9-col MoleculeNet + 31-col RDKit extra = 40-col atom block", baseline_variant="x9_plus_extra_only",
    augmented_prefix="x9_plus_extra_plus_",
    eval_script="eval_graphpfn_pooler_on_fsmol_test_x9_features.py",
    merge_script="merge_x9_features_shards.py",
)


def bootstrap(recipe: Recipe, paper_dir: Path) -> SimpleNamespace:
    """Load the pooler code from <paper_dir>/dev_prior_final and whatever the recipe needs on top.
    Returns env with .bt / .ef / .afe / .atom_features / .robust_mol (None unless the recipe parses SMILES)."""
    if not paper_dir.exists():
        raise SystemExit(f"--paper-dir {paper_dir} does not exist.")
    ns = pooler_glue.load(paper_dir)
    return SimpleNamespace(
        bt=ns.bt, ef=ns.ef, afe=ns.afe, atom_features=ns.atom_features,
        robust_mol=smiles_repair.robust_mol if recipe.atoms == "x9_plus_extra" else None,
    )


def n_atom_cols(recipe: Recipe, env) -> int:
    """Atom-feature width before features_per_group padding."""
    return 32 if recipe.atoms == "native" else 9 + env.afe.N_EXTRA_FEATURES


def read_task(recipe: Recipe, path: Path, *, per_mol_mode: str = "fingerprint", fp_fold: int = 0) -> data.TaskData:
    """Read a task file with everything the recipe needs from disk, including the spliced
    per-molecule matrix. Atom features that need computing are attached by featurize()."""
    task = data.read_task(path, node_features=recipe.atoms == "native", permol=recipe.splice,
                          smiles=recipe.atoms == "x9_plus_extra")
    if recipe.splice:
        task.permol = data.build_permol_features(task.fps, task.descs, per_mol_mode, fp_fold)
        task.fps = task.descs = None          # only the (possibly folded) splice matrix is needed from here on
    return task


def featurize(recipe: Recipe, env, task: data.TaskData, name: str) -> None:
    """Attach the recipe's per-atom features to `task` (no-op for native: the stored node_features are
    used as they are). x9_plus_extra: every mol is built ONCE via robust_mol() and BOTH feature blocks
    are read off that same mol, so molecules with the systematic [PH](=O)(=O)O SMILES defect are
    recovered rather than dropped. Raises UnbuildableTask if a molecule is still unparseable or its
    RDKit atom count disagrees with FS-Mol's node_features count (it would misalign with the stored
    bond graph)."""
    if recipe.atoms == "native":
        return
    from rdkit import Chem

    feats, status = [], []
    for i, (smi, g) in enumerate(zip(task.smiles, task.graphs)):
        n_expected = g["n_atoms"]
        mol, st = env.robust_mol(Chem, smi)
        if mol is None:
            raise UnbuildableTask(f"{name}: molecule {i} unparseable even after robust_mol repair: {smi!r}")
        if mol.GetNumAtoms() != n_expected:
            raise UnbuildableTask(
                f"{name}: molecule {i} atom count {mol.GetNumAtoms()} != FS-Mol node_features "
                f"count {n_expected} for {smi!r} -- would misalign with the stored bond graph."
            )
        try:
            feats.append(env.atom_features.x_plus_extra_from_mol(mol, n_atoms_expected=n_expected))
        except ValueError as e:
            raise UnbuildableTask(f"{name}: molecule {i} ({smi!r}): {e}")
        status.append(st)
    task.atom_feats, task.repair_status = feats, status
    task.smiles = None
