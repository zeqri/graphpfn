"""Zero-shot in-context learning on QM9 using the RELEASED GraphPFN-1.3
checkpoint (the paper's own node-level-trained adapters), for direct
comparison against `qm9_graph_level_icl.py` (this repo's own graph_level-
prior-trained checkpoint) on the exact same QM9 setup.

Uses the released `graphpfn` package's `predict_icl` (src/graphpfn/
inference/icl.py), which loads `GraphPFN.from_pretrained(...)` -- by
default `hf://eremeev-d/graphpfn-1.3/graphpfn-adapters-1_3.pt`, the same
144-tensor adapters-only checkpoint used to warm-start
exp/graphpfn/pretrain/graph_level_warmstart/pretrain.toml. The frozen LimiX
backbone underneath is identical either way.

This is a from-scratch rewrite of tests/qm9_graph_level_icl_virtualnode.py
for this cluster: that script hardcodes JURECA home-directory cache paths
(/p/home/jusers/.../jureca/...) that don't exist here, and its own defaults
(--n-context 1024 --n-query 256) don't match qm9_graph_level_icl.py's
(--n-context 100 --n-query 100). This script defaults to the JUWELS project
paths and the SAME n-context/n-query/benchmark-target as
qm9_graph_level_icl.py, so a plain, unqualified run of each is directly
comparable.

Node/mask construction mirrors qm9_graph_level_icl.py exactly: one virtual
node per molecule, star-wired to only its own atoms; atoms never appear in
train_mask or test_mask (never context, never scored), matching the
"only virtual nodes are ever labeled" training convention -- see that
script's docstring, and the conversation note confirming
tests/qm9_graph_level_icl_virtualnode.py already did this correctly via
mask index ranges rather than an explicit labeled_mask tensor.

Usage (run with cwd=paper/, inside a GPU allocation):
    python -m bin.graphpfn.qm9_released_icl --benchmark-target gap
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# All model/dataset downloads go under the project cache, not the (quota-
# limited) home directory -- must be set before huggingface_hub/torch are
# imported anywhere. Reuses the same cache already populated by
# convert_released_adapters_checkpoint.py (LimiX-16M.ckpt,
# graphpfn-adapters-1_3.pt), so this should not need to download anything.
_CHECKPOINT_DIR = "/p/project1/profound/al-zeqri1/PFN/graphpfn/paper/checkpoints"
os.environ.setdefault("HF_HOME", os.path.join(_CHECKPOINT_DIR, ".cache", "huggingface"))
os.environ.setdefault("TORCH_HOME", os.path.join(_CHECKPOINT_DIR, ".cache", "torch"))
os.environ.setdefault("GRAPHPFN_CHECKPOINT_DIR", _CHECKPOINT_DIR)

import numpy as np

DEFAULT_QM9_ROOT = Path("/p/project1/profound/al-zeqri1/PFN/graphpfn/data/QM9")

# QM9 target names, in the order of data.y columns (torch_geometric convention).
QM9_TARGETS = [
    "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve",
    "u0", "u298", "h298", "g298", "cv",
    "u0_atom", "u298_atom", "h298_atom", "g298_atom", "a", "b", "c",
]

# Target column, unit multiplier, and unit label -- identical convention to
# qm9_graph_level_icl.py, so MAE/RMSE are directly comparable across scripts.
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
    parser.add_argument("--qm9-root", type=Path, default=DEFAULT_QM9_ROOT,
                        help="PyG QM9 root directory (contains raw/ and processed/).")
    parser.add_argument("--benchmark-target", default="gap", choices=sorted(BENCHMARK_TARGETS))
    parser.add_argument("--n-context", type=int, default=100,
                        help="Labeled context molecules. Matches "
                             "qm9_graph_level_icl.py's default so the two "
                             "scripts are directly comparable, rather than "
                             "this checkpoint's own release-script default (1024).")
    parser.add_argument("--n-query", type=int, default=100,
                        help="Query molecules being predicted (this script's 'test set').")
    parser.add_argument("--n-members", type=int, default=10,
                        help="Ensemble members (forward passes). 1 = fastest.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seeds which random QM9 subsample is drawn (there is no "
                             "official QM9 train/test split shipped by PyG's loader).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default=None,
                        help="Optional local path to the GraphPFN adapters checkpoint "
                             "(defaults to the released hf://eremeev-d/graphpfn-1.3 one).")
    return parser.parse_args()


def build_virtual_node_dataset(args, target_idx: int, unit_multiplier: float):
    """Sample molecules and build one GraphDataset (released package's type)
    in the virtual-node formulation -- one labeled virtual node per
    molecule, connected only to its own atoms; atoms never appear in
    train_mask/test_mask (never context, never scored). See module
    docstring.

    Returns (dataset, feature_fit_mask, unit_label). feature_fit_mask marks
    real atom nodes, so `predict_icl` fits feature normalizers on
    representative (non-placeholder) rows -- see its own docstring.
    """
    import dgl
    import torch
    from torch_geometric.datasets import QM9

    from graphpfn.data import GraphDataset

    full = QM9(root=str(args.qm9_root))
    unit_label = BENCHMARK_TARGETS[args.benchmark_target][2]
    print(f"Loaded QM9: {len(full)} molecules, target = {args.benchmark_target!r} "
          f"(QM9 column {QM9_TARGETS[target_idx]!r}, unit {unit_label})")

    n_total = args.n_context + args.n_query
    assert n_total <= len(full), (
        f"requested {n_total} molecules exceeds QM9 size ({len(full)})"
    )

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(full))[:n_total]
    mols = [full[int(i)] for i in perm]

    atom_feat_dim = mols[0].x.shape[1]

    atom_features = []
    edge_src_parts = []
    edge_dst_parts = []
    vsrc_parts = []
    vdst_parts = []
    graph_targets = np.zeros(n_total, dtype=np.float32)

    offset = 0
    n_atom_nodes = sum(int(mol.x.shape[0]) for mol in mols)
    for i, mol in enumerate(mols):
        n_atoms = int(mol.x.shape[0])
        atom_features.append(mol.x.numpy().astype(np.float32))

        ei = mol.edge_index.numpy()
        edge_src_parts.append(ei[0] + offset)
        edge_dst_parts.append(ei[1] + offset)

        vid = n_atom_nodes + i
        atoms = np.arange(offset, offset + n_atoms)
        vsrc_parts.append(np.full(n_atoms, vid, dtype=atoms.dtype))
        vdst_parts.append(atoms)

        graph_targets[i] = float(mol.y[0, target_idx]) * unit_multiplier
        offset += n_atoms

    n_nodes = n_atom_nodes + n_total

    features = np.concatenate(
        atom_features + [np.zeros((n_total, atom_feat_dim), dtype=np.float32)],
        axis=0,
    )

    edge_src = np.concatenate(edge_src_parts + vsrc_parts)
    edge_dst = np.concatenate(edge_dst_parts + vdst_parts)
    graph = dgl.graph(
        data=(edge_src, edge_dst),
        num_nodes=n_nodes,
        idtype=torch.int32,
    )

    node_targets = np.zeros(n_nodes, dtype=np.float32)
    node_targets[n_atom_nodes:] = graph_targets

    feature_fit_mask = np.zeros(n_nodes, dtype=bool)
    feature_fit_mask[:n_atom_nodes] = True  # real atom nodes only

    train_mask = np.zeros(n_nodes, dtype=bool)  # context virtual nodes
    test_mask = np.zeros(n_nodes, dtype=bool)  # query virtual nodes
    train_mask[n_atom_nodes : n_atom_nodes + args.n_context] = True
    test_mask[n_atom_nodes + args.n_context :] = True

    dataset = GraphDataset(
        name=f"qm9-released-icl-{args.benchmark_target}",
        graph=graph,
        features={"other": features},
        targets=node_targets,
        masks={
            "train": train_mask,
            "val": np.zeros(n_nodes, dtype=bool),
            "test": test_mask,
        },
        task_type="regression",
    )

    print(f"Disjoint union: {n_atom_nodes} atoms + {n_total} virtual (graph) nodes, "
          f"{graph.num_edges()} directed edges "
          f"({train_mask.sum()} context / {test_mask.sum()} query graphs)")

    return dataset, feature_fit_mask, unit_label


def main() -> None:
    args = parse_args()

    import torch

    from graphpfn.inference.icl import predict_icl
    from graphpfn.inference.util import compute_metrics

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    target_idx, unit_multiplier, unit_label = BENCHMARK_TARGETS[args.benchmark_target]
    dataset, feature_fit_mask, unit_label = build_virtual_node_dataset(
        args, target_idx, unit_multiplier
    )

    model_kwargs = {"ensemble_kwargs": {"n_members": args.n_members}}
    if args.checkpoint is not None:
        model_kwargs["checkpoint"] = args.checkpoint

    device = torch.device(args.device)
    node_preds = predict_icl(
        dataset,
        model_kwargs=model_kwargs,
        feature_fit_mask=feature_fit_mask,
        amp=(device.type == "cuda"),
        device=device,
    )

    test_mask = dataset.masks["test"]
    train_mask = dataset.masks["train"]

    metrics = compute_metrics(
        y_true=dataset.targets[test_mask],
        y_pred=node_preds[test_mask],
        task_type="regression",
    )

    context_mean = float(dataset.targets[train_mask].mean())
    baseline_mae = float(np.abs(dataset.targets[test_mask] - context_mean).mean())

    print(f"\n=== Released GraphPFN-1.3 (paper's node-level-trained adapters, "
          f"zero-shot ICL, virtual-node) on QM9 [{args.benchmark_target}] ===")
    print(f"context molecules: {args.n_context}, query molecules: {args.n_query}, "
          f"ensemble members: {args.n_members}, seed: {args.seed}")
    print(f"R2:   {metrics['r2']:.4f}")
    print(f"MAE:  {metrics['mae']:.4f} {unit_label}")
    print(f"MAE of constant (context-mean) baseline: {baseline_mae:.4f} {unit_label}")

    print(
        "\nNOTE: zero-shot ICL, frozen checkpoint, no gradient steps taken. These "
        "adapters were trained on the node-level prior (dense, single connected "
        "graphs, no supernode), so the virtual-node star topology here is off their "
        "training distribution -- unlike qm9_graph_level_icl.py's checkpoint, which "
        "was trained directly on this topology. context/query molecules are a "
        f"random {args.n_context + args.n_query}-molecule subsample (seeded by "
        "--seed), not an official QM9 split. Run qm9_graph_level_icl.py with the "
        "same --benchmark-target/--n-context/--n-query/--seed for a direct comparison."
    )


if __name__ == "__main__":
    main()
