"""Command-line pieces shared by the trainers, evaluators and merge scripts, plus the default
locations (relative to this file, no absolute paths)."""

from __future__ import annotations

import argparse
from pathlib import Path

PAPER_DIR_DEFAULT = Path(__file__).resolve().parents[2]                      # fsmol_core/ -> fs_mol/ -> paper/
FSMOL_DIR_DEFAULT = PAPER_DIR_DEFAULT.parent / "datasets" / "fs-mol"        # train/ valid/ test/ + fsmol-0.1.json + target_info.csv


def parse_int_list(spec: str) -> list[int]:
    out = sorted({int(p) for p in spec.replace(" ", "").split(",") if p})
    if not out:
        raise argparse.ArgumentTypeError("empty list")
    return out


def add_data_args(p: argparse.ArgumentParser, splits: str) -> None:
    """--paper-dir / --fsmol-dir / --fsmol-data / --task-list-json / --pooler-checkpoint."""
    p.add_argument("--paper-dir", type=Path, default=PAPER_DIR_DEFAULT,
                   help=f"'paper' dir containing dev_prior_final/, lib/, vendor/. Default: {PAPER_DIR_DEFAULT}")
    p.add_argument("--fsmol-dir", type=Path, default=FSMOL_DIR_DEFAULT,
                   help=f"FS-Mol data dir (train/ valid/ test/, fsmol-0.1.json, target_info.csv). Default: {FSMOL_DIR_DEFAULT}")
    p.add_argument("--fsmol-data", type=Path, default=None, help=f"Dir with {splits} *.jsonl.gz (default: <fsmol-dir>).")
    p.add_argument("--task-list-json", type=Path, default=None, help="Default: <fsmol-dir>/fsmol-0.1.json.")


def add_edge_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--keep-edge-direction", action="store_true",
                   help="Do NOT add reverse edges (default: FS-Mol's stored bonds are made bidirectional).")
