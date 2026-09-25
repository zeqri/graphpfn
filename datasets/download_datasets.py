"""Download + process the datasets the evaluation scripts use. Run ONCE on a machine with internet
(e.g. a login node) before submitting jobs -- compute nodes have none, and torch_geometric otherwise
tries to download a missing dataset at load time.

  * MoleculeNet (bace, bbbp, clintox, esol, freesolv, lipo, sider) -> <root>/moleculenet/<name>/{raw,processed}
  * ZINC-12k (subset=True; train / val / test)                        -> <root>/zinc/{raw,subset/processed}
  * AQSOL (train / val / test)                                        -> <root>/aqsol/{raw,processed}

<root> defaults to this directory (<repo>/datasets), which is where the evaluation scripts look.
Datasets already on disk are loaded, not re-downloaded. Everything runs sequentially, so no two
processes race on the same directory. NOT downloaded here: the scaffold splits
(moleculenet/<name>/split/scaffold_split.json) and aqsol/data_curated.csv, which ship with the repo.

After downloading, each MoleculeNet CSV's row count is checked against every embedding model's
<dataset>_meta.json "shape" -- the embeddings' split_idx index rows of the CSV they were computed
from, so a mismatch means the downloaded CSV is not that file.

Usage:
    python download_datasets.py                                   # everything, into this directory
    python download_datasets.py --datasets bbbp clintox lipo sider aqsol
    python download_datasets.py --datasets zinc --root /some/other/datasets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch_geometric.datasets import AQSOL, ZINC, MoleculeNet

DATASETS_DIR = Path(__file__).resolve().parent
REPO_DIR = DATASETS_DIR.parent
MOLECULENET = ["bace", "bbbp", "clintox", "esol", "freesolv", "lipo", "sider"]
ALL_DATASETS = [*MOLECULENET, "zinc", "aqsol"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=ALL_DATASETS, default=ALL_DATASETS,
                        help="Which datasets to download (default: all).")
    parser.add_argument("--root", type=Path, default=DATASETS_DIR,
                        help="Where to save them; holds moleculenet/, zinc/, aqsol/ (default: %(default)s).")
    parser.add_argument("--embeddings-root", type=Path, default=REPO_DIR / "embeddings",
                        help="Root holding <model>/<dataset>/<dataset>_meta.json, for the row-count check "
                        "(default: %(default)s).")
    return parser.parse_args()


def csv_row_count(path: Path) -> int:
    """Data rows the way MoleculeNet.process counts them: non-empty lines after the header."""
    with open(path) as f:
        return len([x for x in f.read().split("\n")[1:-1] if len(x) > 0])


def check_against_embeddings(ds: MoleculeNet, name: str, embeddings_root: Path) -> bool:
    """True iff the raw CSV has as many rows as every embedding meta's "shape"[0]."""
    n_rows = csv_row_count(Path(ds.raw_paths[0]))
    ok = True
    for meta_path in sorted(embeddings_root.glob(f"*/{name}/{name}_meta.json")):
        expected = json.loads(meta_path.read_text())["shape"][0]
        if n_rows != expected:
            print(f"  [MISMATCH] {ds.raw_paths[0]} has {n_rows} rows, {meta_path} expects {expected}")
            ok = False
    if ok:
        print(f"  CSV rows = {n_rows}, consistent with the embeddings' meta")
    return ok


def main() -> None:
    args = parse_args()
    root = args.root
    all_ok = True

    for name in args.datasets:
        print(f"\n=== {name} ===")
        if name in MOLECULENET:
            ds = MoleculeNet(root=str(root / "moleculenet"), name=name)
            print(f"  {len(ds)} molecules in {root / 'moleculenet' / name}")
            all_ok &= check_against_embeddings(ds, name, args.embeddings_root)
        elif name == "zinc":
            for split in ("train", "val", "test"):
                ds = ZINC(root=str(root / "zinc"), subset=True, split=split)
                print(f"  {split}: {len(ds)} molecules")
        else:
            for split in ("train", "val", "test"):
                ds = AQSOL(root=str(root / "aqsol"), split=split)
                print(f"  {split}: {len(ds)} graphs")
            if not (root / "aqsol" / "data_curated.csv").exists():
                print(f"  [warn] {root / 'aqsol' / 'data_curated.csv'} missing -- eval_aqsol.py needs it for atom features")

    print("\nDone." if all_ok else "\nDone, but some MoleculeNet CSVs do not match the embeddings (see [MISMATCH] above).")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
