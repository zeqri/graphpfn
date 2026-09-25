"""Helpers shared by the scaffold-split encoders. A split's indices are rows of ONE specific raw CSV,
so (1) refuse to encode against any other file (e.g. a re-downloaded CSV with an extra or reordered
row, which would silently pair every embedding with the wrong molecule) and (2) read the SMILES with
exactly the parser datasets/make_scaffold_splits.py and torch_geometric's MoleculeNet use, so row r
means the same molecule everywhere."""

import hashlib
import json
import re
from pathlib import Path

from torch_geometric.datasets import MoleculeNet


def read_csv_smiles(csv_path: Path, name: str) -> tuple[list[str], str]:
    """(SMILES per CSV data row, SMILES column header) for MoleculeNet dataset `name`, parsed like
    MoleculeNet.process: non-empty lines after the header, quoted fields stripped, SMILES taken from
    MoleculeNet.names[name]'s column index (the header name differs between files, e.g. bace's 'mol')."""
    col = MoleculeNet.names[name][3]
    with open(csv_path) as f:
        text = f.read().split('\n')
    lines = [x for x in text[1:-1] if len(x) > 0]
    header = re.sub(r'\".*\"', '', text[0]).split(',')[col]
    return [re.sub(r'\".*\"', '', line).split(',')[col] for line in lines], header


def check_split_source(csv_path: Path, split_json: Path) -> str:
    """Returns the CSV's sha256 after checking it against the split_info.json written next to
    split_json by datasets/make_scaffold_splits.py; raises if they differ or split_info.json is missing."""
    info_path = Path(split_json).with_name('split_info.json')
    if not info_path.exists():
        raise FileNotFoundError(f'{info_path} missing -- regenerate the split with datasets/make_scaffold_splits.py')
    info = json.loads(info_path.read_text())
    digest = hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()
    if digest != info['source_csv_sha256']:
        raise RuntimeError(
            f'{csv_path} (sha256 {digest[:12]}...) is not the CSV {split_json} was built from '
            f'(sha256 {info["source_csv_sha256"][:12]}...) -- regenerate the split or restore that CSV.'
        )
    return digest
