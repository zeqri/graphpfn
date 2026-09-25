"""Scaffold-split the MoleculeNet datasets (80/10/10) directly in RAW-CSV-ROW index space.

Why: the old splits were built from torch_geometric's MoleculeNet list (graphs/ogb/
scaffold_split_moleculenet.py). PyG drops molecules RDKit cannot parse and renumbers the rest, so for
any dataset with an unparseable row, PyG index i != CSV row i from that row on. The embedding scripts
and the evaluation (common.csv_row_to_pyg_index) read split indices as CSV rows, so such a split is
applied to the wrong molecules and scaffolds leak across train/valid/test (BBBP: 68 shared train/test
scaffolds). ClinTox was already patched this way (scaffold_split_clintox.py); this script does it for
every dataset, so there is one index space everywhere: CSV data rows (0-indexed, header excluded).

Algorithm (unchanged): greedy Bemis-Murcko scaffold split (no chirality), scaffold groups sorted by
(size, first index) descending, largest groups to train first -- the OGB / Hu et al. convention.
Unparseable rows are left out of every split. Rows are read with the same line parser PyG uses, so
row r here is row r for csv_row_to_pyg_index.

Outputs, per dataset under <out-root>/<name>/split/:
  scaffold_split.json  {"train": [...], "valid": [...], "test": [...]}  -- CSV rows (same schema as before,
                       so the embedding scripts read it unchanged)
  split_info.json      provenance: index space, source CSV file + sha256 + row count, unparseable rows,
                       and the scaffold-overlap check (always 0)

--compare-root DIR additionally reports, per dataset, whether the new split equals DIR/<name>/split/
scaffold_split.json, and how many scaffolds that old split shares across train/valid/test when read as
CSV rows.

Usage:
    python make_scaffold_splits.py                                  # all datasets, repo raw CSVs
    python make_scaffold_splits.py --csv-root /path/to/moleculenet --compare-root ../datasets/moleculenet
    python make_scaffold_splits.py --datasets bbbp clintox
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.datasets import MoleculeNet

RDLogger.DisableLog("rdApp.*")

DATASETS_NEW_DIR = Path(__file__).resolve().parent
REPO_DIR = DATASETS_NEW_DIR.parent
ALL_DATASETS = ["bace", "bbbp", "clintox", "esol", "freesolv", "lipo", "sider"]
SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=ALL_DATASETS, default=ALL_DATASETS,
                        help="Which datasets to split (default: all).")
    parser.add_argument("--csv-root", type=Path, default=REPO_DIR / "datasets" / "moleculenet",
                        help="MoleculeNet root holding <name>/raw/<csv> (default: %(default)s).")
    parser.add_argument("--out-root", type=Path, default=DATASETS_NEW_DIR / "moleculenet",
                        help="Where to write <name>/split/ (default: %(default)s).")
    parser.add_argument("--compare-root", type=Path, default=None,
                        help="Optional root with existing <name>/split/scaffold_split.json to compare against.")
    parser.add_argument("--frac-train", type=float, default=0.8)
    parser.add_argument("--frac-valid", type=float, default=0.1)
    return parser.parse_args()


def read_csv_smiles(csv_path: Path, name: str) -> list[str]:
    """SMILES per CSV data row, parsed exactly like torch_geometric's MoleculeNet.process (and
    therefore like common.csv_row_to_pyg_index), so row numbers agree everywhere."""
    smiles_col = MoleculeNet.names[name][3]
    with open(csv_path) as f:
        lines = [x for x in f.read().split("\n")[1:-1] if len(x) > 0]
    return [re.sub(r"\".*\"", "", line).split(",")[smiles_col] for line in lines]


def generate_scaffold(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_split(scaffolds: list[str | None], frac_train: float, frac_valid: float) -> tuple[dict, list[int]]:
    """({train|valid|test: CSV rows}, unparseable rows) -- greedy scaffold split over parseable rows."""
    groups: dict[str, list[int]] = defaultdict(list)
    unparsable = []
    for row, scaffold in enumerate(scaffolds):
        if scaffold is None:
            unparsable.append(row)
        else:
            groups[scaffold].append(row)

    scaffold_sets = [s for _, s in sorted(groups.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)]
    n = len(scaffolds) - len(unparsable)
    train_cutoff = frac_train * n
    valid_cutoff = (frac_train + frac_valid) * n

    split = {s: [] for s in SPLITS}
    for scaffold_set in scaffold_sets:
        if len(split["train"]) + len(scaffold_set) > train_cutoff:
            if len(split["train"]) + len(split["valid"]) + len(scaffold_set) > valid_cutoff:
                split["test"] += scaffold_set
            else:
                split["valid"] += scaffold_set
        else:
            split["train"] += scaffold_set
    return split, unparsable


def scaffold_overlap(split: dict, scaffolds: list[str | None]) -> dict[str, int]:
    """Scaffolds shared between splits when split indices are read as CSV rows (0 for a valid split)."""
    sets = {s: {scaffolds[r] for r in split[s] if r < len(scaffolds)} - {None} for s in SPLITS}
    return {
        "train_test": len(sets["train"] & sets["test"]),
        "train_valid": len(sets["train"] & sets["valid"]),
        "valid_test": len(sets["valid"] & sets["test"]),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    for name in args.datasets:
        csv_name = MoleculeNet.names[name][1].removesuffix(".gz")
        csv_path = args.csv_root / name / "raw" / csv_name
        scaffolds = [generate_scaffold(s) for s in read_csv_smiles(csv_path, name)]
        split, unparsable = scaffold_split(scaffolds, args.frac_train, args.frac_valid)
        overlap = scaffold_overlap(split, scaffolds)
        assert not any(overlap.values()), f"{name}: scaffold overlap {overlap}"

        out_dir = args.out_root / name / "split"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "scaffold_split.json", "w") as f:
            json.dump(split, f)
        with open(out_dir / "split_info.json", "w") as f:
            json.dump({
                "index_space": "csv_row -- 0-indexed data rows of source_csv (header excluded), parsed like "
                               "torch_geometric MoleculeNet.process",
                "source_csv": csv_name,
                "source_csv_sha256": sha256(csv_path),
                "n_csv_rows": len(scaffolds),
                "unparsable_rows": unparsable,
                "split_sizes": {s: len(split[s]) for s in SPLITS},
                "method": f"greedy Bemis-Murcko scaffold split, no chirality, "
                          f"{args.frac_train:g}/{args.frac_valid:g}/{1 - args.frac_train - args.frac_valid:g}",
                "scaffold_overlap": overlap,
            }, f, indent=2)

        msg = (f"{name:9s} rows={len(scaffolds):5d} unparsable={len(unparsable):2d} "
               f"sizes={[len(split[s]) for s in SPLITS]}")
        if args.compare_root is not None:
            old = json.load(open(args.compare_root / name / "split" / "scaffold_split.json"))
            same = all(old[s] == split[s] for s in SPLITS)
            msg += f" | identical to old: {same}"
            if not same:
                msg += f" (old read as CSV rows: sizes={[len(old[s]) for s in SPLITS]}, " \
                       f"shared scaffolds={scaffold_overlap(old, scaffolds)})"
        print(msg)
        print(f"          -> {out_dir}")


if __name__ == "__main__":
    main()
