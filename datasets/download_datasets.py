"""Download + process the datasets the training, evaluation and embedding scripts use. Run it once,
before make_scaffold_splits.py.

  * MoleculeNet (bace, bbbp, clintox, esol, freesolv, lipo, sider) -> <root>/moleculenet/<name>/{raw,processed}
  * ZINC-12k (subset=True; train / val / test)                        -> <root>/zinc/{raw,subset/processed}
  * AQSOL (train / val / test)                                        -> <root>/aqsol/{raw,processed}

<root> defaults to this directory (<repo>/datasets), which is where the other scripts look.
Datasets already on disk are loaded, not re-downloaded. Everything runs sequentially, so no two
processes race on the same directory. NOT downloaded here: aqsol/data_curated.csv, which ships with
the repo. The scaffold splits are built from the downloaded CSVs by make_scaffold_splits.py.

After downloading, each MoleculeNet raw CSV's sha256 is checked against EXPECTED_CSV_SHA256, the files
the paper's splits and results were computed from. The scaffold split indices are rows of that exact
file, so a different CSV (e.g. an upstream re-release with a row added or reordered) would give
different splits and results.

Usage:
    python download_datasets.py                                   # everything, into this directory
    python download_datasets.py --datasets bbbp clintox lipo sider aqsol
    python download_datasets.py --datasets zinc --root /some/other/datasets
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from torch_geometric.datasets import AQSOL, ZINC, MoleculeNet

DATASETS_DIR = Path(__file__).resolve().parent
MOLECULENET = ["bace", "bbbp", "clintox", "esol", "freesolv", "lipo", "sider"]
ALL_DATASETS = [*MOLECULENET, "zinc", "aqsol"]

# sha256 of each MoleculeNet raw CSV the paper's scaffold splits and embeddings were computed from.
EXPECTED_CSV_SHA256 = {
    "bace": "f3fb9ce90bada3e2bd6148b0df13f8f8145a357bf87df0dd5b391ede974fc737",
    "bbbp": "d07a38487aeac5cee5508413e468043ef3097451d2a112701c2d60be9ec6b662",
    "clintox": "9999816e760dd838358b5d88e81cea2fc062be4458ffa412ceecdba4f88a67b6",
    "esol": "8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189",
    "freesolv": "ab5895d914ee87cb563bd7b9611e869527bba45bec6b014d34dc495a0f9dcb72",
    "lipo": "aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd",
    "sider": "71efc6ac4ca82d6545bc512509863281be0f17234afd7adaf71a71899988302e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=ALL_DATASETS, default=ALL_DATASETS,
                        help="Which datasets to download (default: all).")
    parser.add_argument("--root", type=Path, default=DATASETS_DIR,
                        help="Where to save them; holds moleculenet/, zinc/, aqsol/ (default: %(default)s).")
    return parser.parse_args()


def check_csv_sha256(ds: MoleculeNet, name: str) -> bool:
    """True iff the raw CSV is byte-identical to the one the paper used."""
    path = Path(ds.raw_paths[0])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_CSV_SHA256[name]:
        print(f"  [MISMATCH] {path} has sha256 {digest[:12]}..., expected {EXPECTED_CSV_SHA256[name][:12]}...")
        return False
    print(f"  sha256 {digest[:12]}... matches the paper's CSV")
    return True


def main() -> None:
    args = parse_args()
    root = args.root
    all_ok = True

    for name in args.datasets:
        print(f"\n=== {name} ===")
        if name in MOLECULENET:
            ds = MoleculeNet(root=str(root / "moleculenet"), name=name)
            print(f"  {len(ds)} molecules in {root / 'moleculenet' / name}")
            all_ok &= check_csv_sha256(ds, name)
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

    print("\nDone." if all_ok else "\nDone, but some MoleculeNet CSVs differ from the paper's (see [MISMATCH] above).")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
