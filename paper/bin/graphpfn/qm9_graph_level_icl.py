"""Zero-shot in-context learning on QM9 using the graph_level checkpoint.

Loads the EMA weights from the currently running/most recent graph_level
pretraining checkpoint (`exp/graphpfn/pretrain/graph_level/pretrain` by
default) and runs a single ICL forward pass on real QM9 molecules, using the
SAME virtual-node convention the graph_level prior was trained on: one
placeholder "virtual" node per molecule, star-wired to only that molecule's
own atoms, with the molecule's target sitting on the virtual node and atoms
carrying no label (see lib/graphpfn/prior/priors/graph_level.py). A subset of
molecules is used as labeled ICL context; the rest are queried -- no gradient
step is taken.

Unlike tests/qm9_graph_level_icl_virtualnode.py (which uses the released
graphpfn package's `predict_icl` against the released/HF-Hub checkpoint),
this script talks directly to the internal research model
(lib.graphpfn.model.GraphPFN) and reuses this session's just-fixed
evaluation path (lib.graphpfn.prior.util.convert_to_graph_dataset +
bin.graphpfn.pretrain.evaluate_dataset) -- so the QM9 masks/feature-fit/
label-standardization handling exactly matches what periodic synthetic eval
already does for this prior, rather than the released package's (different)
assumptions.

Node layout mirrors `graph_level.py`'s own convention exactly: [context
virtual nodes][query virtual nodes][atoms], so `convert_to_graph_dataset`'s
prefix-based "train = [0, n_train_nodes)" logic identifies context/query
correctly.

Usage (run with cwd=paper/, inside a GPU allocation -- see run_full.sh for
the module-load / venv-activate incantation):
    python -m bin.graphpfn.qm9_graph_level_icl --benchmark-target gap
"""

from __future__ import annotations

import argparse
import random
import tomllib
from pathlib import Path

import numpy as np
import torch

import lib
from bin.graphpfn.graph_level_label_standardization_validation import load_ema_model
from bin.graphpfn.pretrain import evaluate_dataset
from lib.graphpfn.prior.prior_typings import PriorDataset
from lib.graphpfn.prior.util import convert_to_graph_dataset
from lib.util import TaskType

DEFAULT_TOML_PATH = Path("exp/graphpfn/pretrain/graph_level/pretrain.toml")
DEFAULT_OUTPUT_DIR = Path("exp/graphpfn/pretrain/graph_level/pretrain")
# Repo root's data/QM9 (one level up from paper/, where this script runs
# from) -- already downloaded and processed; NOT paper/data/QM9.
DEFAULT_QM9_ROOT = Path("/p/project1/profound/al-zeqri1/PFN/graphpfn/data/QM9")

# QM9 target names, in the order of data.y columns (torch_geometric convention).
QM9_TARGETS = [
    "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve",
    "u0", "u298", "h298", "g298", "cv",
    "u0_atom", "u298_atom", "h298_atom", "g298_atom", "a", "b", "c",
]

# Target column, unit multiplier (applied to PyG's raw y value), and unit
# label, matching the convention used by tests/qm9_graph_level_icl_virtualnode.py
# (see that file's docstring for the Hartree->eV / x1000 rationale) so
# results are directly comparable across scripts.
BENCHMARK_TARGETS: dict[str, tuple[int, float, str]] = {
    "mu": (0, 1.0, "D"),
    "alpha": (1, 1.0, "bohr^3"),
    "homo": (2, 1000.0, "meV"),
    "lumo": (3, 1000.0, "meV"),
    "gap": (4, 1000.0, "meV"),
    "r2": (5, 1.0, "bohr^2"),
    "zpve": (6, 1000.0, "meV"),
    "cv": (11, 1.0, "cal/mol K"),
    "u0": (12, 1000.0, "meV"),  # u0_atom (atomization)
    "u": (13, 1000.0, "meV"),  # u298_atom (atomization)
    "h": (14, 1000.0, "meV"),  # h298_atom (atomization)
    "g": (15, 1000.0, "meV"),  # g298_atom (atomization)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toml", type=Path, default=DEFAULT_TOML_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory containing checkpoint.pt (the graph_level run's output dir).")
    parser.add_argument("--qm9-root", type=Path, default=DEFAULT_QM9_ROOT,
                        help="PyG QM9 root directory (contains raw/ and processed/).")
    parser.add_argument("--benchmark-target", default="gap", choices=sorted(BENCHMARK_TARGETS))
    parser.add_argument("--n-context", type=int, default=100,
                        help="Labeled context molecules (virtual nodes with real "
                             "targets attended to via ICL). Kept modest by default "
                             "so total node count stays close to the graph_level "
                             "prior's own training budget (total_n_nodes_budget, "
                             "1000-5000 nodes) -- QM9 molecules average ~18 atoms.")
    parser.add_argument("--n-query", type=int, default=100,
                        help="Query molecules being predicted (this script's 'test set').")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seeds which random QM9 subsample is drawn (there is no "
                             "official QM9 train/test split shipped by PyG's loader).")
    args = parser.parse_args()
    return args


def load_qm9_molecules(qm9_root: Path, n_total: int, seed: int):
    from torch_geometric.datasets import QM9

    full = QM9(root=str(qm9_root))
    print(f"Loaded QM9: {len(full)} molecules")
    assert n_total <= len(full), f"requested {n_total} molecules exceeds QM9 size ({len(full)})"

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(full))[:n_total]
    return [full[int(i)] for i in perm]


def build_qm9_prior_dataset(mols: list, n_context: int, target_idx: int, unit_multiplier: float) -> PriorDataset:
    """Builds the [context virtual][query virtual][atoms] layout, exactly
    mirroring graph_level.py's `_build_disjoint_union` convention: the first
    n_context virtual-node indices are context, the rest of the first
    n_graphs indices are query, and real atom nodes follow. `mols` is
    assumed to already be in random order (see `load_qm9_molecules`), so
    `mols[:n_context]` / `mols[n_context:]` is itself a random context/query
    split -- no additional shuffling needed.
    """
    n_graphs = len(mols)
    atom_feat_dim = mols[0].x.shape[1]

    edge_src_parts: list[np.ndarray] = []
    edge_dst_parts: list[np.ndarray] = []
    atom_features_parts: list[np.ndarray] = []
    graph_targets = np.zeros(n_graphs, dtype=np.float32)

    atom_offset = n_graphs  # atoms start right after the n_graphs virtual nodes
    for i, mol in enumerate(mols):
        n_atoms = int(mol.x.shape[0])
        atom_features_parts.append(mol.x.numpy().astype(np.float32))

        ei = mol.edge_index.numpy()
        edge_src_parts.append(ei[0] + atom_offset)
        edge_dst_parts.append(ei[1] + atom_offset)

        atoms = np.arange(atom_offset, atom_offset + n_atoms)
        virtual_id = np.full(n_atoms, i, dtype=atoms.dtype)
        edge_src_parts.append(virtual_id)
        edge_dst_parts.append(atoms)
        edge_src_parts.append(atoms)
        edge_dst_parts.append(virtual_id)

        graph_targets[i] = float(mol.y[0, target_idx]) * unit_multiplier
        atom_offset += n_atoms

    n_nodes = atom_offset

    features = np.concatenate(
        [np.zeros((n_graphs, atom_feat_dim), dtype=np.float32)] + atom_features_parts,
        axis=0,
    )
    labels = np.zeros(n_nodes, dtype=np.float32)
    labels[:n_graphs] = graph_targets

    edges = np.stack(
        [np.concatenate(edge_src_parts), np.concatenate(edge_dst_parts)], axis=0
    )

    is_virtual = np.zeros(n_nodes, dtype=bool)
    is_virtual[:n_graphs] = True

    print(
        f"Disjoint union: {n_nodes - n_graphs} atoms + {n_graphs} virtual (graph) "
        f"nodes = {n_nodes} total, {edges.shape[1]} directed edges "
        f"({n_context} context / {n_graphs - n_context} query molecules)"
    )

    return PriorDataset(
        features=torch.from_numpy(features),
        labels=torch.from_numpy(labels),
        edges=torch.from_numpy(edges).long(),
        n_train_nodes=n_context,
        task_type=TaskType.REGRESSION,
        labeled_mask=torch.from_numpy(is_virtual),
        feature_fit_mask=torch.from_numpy(~is_virtual),
        # QM9 targets are raw (not pre-standardized anywhere) -- unlike the
        # graph_level prior's own synthetic labels, so evaluate_dataset
        # SHOULD standardize on the context subset here, exactly like a
        # real (GraphLand-style) eval dataset.
        labels_standardized=False,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.toml, "rb") as f:
        toml_config = tomllib.load(f)
    config = toml_config["base_config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = lib.load_checkpoint(args.output_dir, map_location=device)
    print(f"Loaded checkpoint from {args.output_dir} at step={checkpoint['step']}")

    amp_enabled = (
        config.get("amp", False)
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    print(f"device={device} amp_enabled={amp_enabled}")

    graphpfn = load_ema_model(config, checkpoint, device)

    target_idx, unit_multiplier, unit_label = BENCHMARK_TARGETS[args.benchmark_target]
    n_total = args.n_context + args.n_query
    mols = load_qm9_molecules(args.qm9_root, n_total, args.seed)

    prior_dataset = build_qm9_prior_dataset(mols, args.n_context, target_idx, unit_multiplier)
    graph_dataset = convert_to_graph_dataset(prior_dataset, name=f"qm9-{args.benchmark_target}")

    with torch.inference_mode():
        metrics = evaluate_dataset(
            graphpfn, graph_dataset, device=device, amp_enabled=amp_enabled, parts=["test"]
        )

    train_mask = graph_dataset.data["masks"]["train"]
    test_mask = graph_dataset.data["masks"]["test"]
    context_targets = graph_dataset.data["labels"][train_mask]
    query_targets = graph_dataset.data["labels"][test_mask]
    context_mean = float(context_targets.mean())
    baseline_mae = float(np.abs(query_targets - context_mean).mean())

    print(f"\n=== GraphPFN (graph_level checkpoint, step={checkpoint['step']}) "
          f"zero-shot ICL on QM9 [{args.benchmark_target}] ===")
    print(f"context molecules: {args.n_context}, query molecules: {args.n_query}, seed: {args.seed}")
    if "test" not in metrics:
        print(f"\nNo valid predictions (metrics={metrics}) -- likely non-finite model output.")
        return
    test_metrics = metrics["test"]
    print(f"R2:   {test_metrics['r2']:.4f}")
    print(f"MAE:  {test_metrics['mae']:.4f} {unit_label}")
    print(f"RMSE: {test_metrics['rmse']:.4f} {unit_label}")
    print(f"MAE of constant (context-mean) baseline: {baseline_mae:.4f} {unit_label}")
    print(
        "\nNOTE: zero-shot ICL, no gradient steps taken. QM9's virtual-node star "
        "topology matches the graph_level prior's own training convention (unlike "
        "the node-level checkpoint tested in tests/qm9_graph_level_icl_virtualnode.py, "
        "for which this topology is off-prior), but QM9 molecules/features/targets "
        "are real chemistry, not synthetic SBM graphs -- so this measures "
        "out-of-distribution transfer, not in-distribution generalization. "
        "context/query molecules are a random subsample (seeded by --seed), not an "
        "official QM9 split (PyG's QM9 loader does not ship one)."
    )


if __name__ == "__main__":
    main()
