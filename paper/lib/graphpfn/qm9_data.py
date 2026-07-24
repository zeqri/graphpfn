"""Real QM9 molecule loading for graph-level periodic evaluation.

Promoted from `dev/pool_icl_real_limix.py`'s `sample_one_qm9_dataset`, with
one change: rather than returning per-molecule Python lists (which that
script's own per-molecule loop needed), this returns the same
(atom_features, graph, y_per_molecule) batched-tensor shape
`lib.graphpfn.prior.graph_level.sample_graph_level_dataset` returns for the
synthetic prior, so both feed `GraphLevelGraphPFN.forward` identically. Real
bond distances come from QM9's actual 3D conformer coordinates (`data.pos`),
not a fitted prior -- the same mechanism `EdgeDistanceEncoder` consumes
elsewhere.

Kept in its own module (rather than inlined in the pretrain script) because
the deferred full-QM9-finetuning follow-up will need this exact same loading
code too.
"""

from __future__ import annotations

import dgl
import numpy as np
import torch
from torch_geometric.datasets import QM9

QM9_TARGETS = [
    "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", "u298", "h298",
    "g298", "cv", "u0_atom", "u298_atom", "h298_atom", "g298_atom", "A", "B", "C",
]


def load_qm9(root: str) -> QM9:
    return QM9(root=root)


def sample_one_qm9_dataset(
    dataset: QM9,
    n_graphs_range: tuple[int, int],
    target_idx: int,
    rng: np.random.Generator,
    all_pairs: bool = False,
) -> tuple[torch.Tensor, dgl.DGLGraph, torch.Tensor]:
    """Draw a random subset of real QM9 molecules as one eval "dataset".

    Returns (atom_features, graph, y_per_molecule) -- same shape convention
    as `sample_graph_level_dataset`, so the caller's context/query split +
    standardization + GraphLevelGraphPFN.forward call is identical for
    synthetic and real data.

    `all_pairs=False` (default): `graph`'s edges are the real chemical bonds
    (`data.edge_index`) -- geometric attention only ever sees 1-hop bonded
    distances.

    `all_pairs=True`: `graph` is fully connected within each molecule (every
    ordered atom pair i != j, zero cross-molecule edges as always), with
    `edata["distance"]` set from real 3D coordinates for every pair, not just
    bonded ones. This tests whether restricting geometric attention to 1-hop
    bonds (rather than the complete pairwise-distance set EGNN's own QM9
    setup uses, see Satorras et al. 2021 Section 3.3/5.3) is what caps
    whole-molecule-shape-dependent targets (A/B/C, r2) -- since a complete
    distance matrix is a lossless E(n)-invariant description of geometry
    (ibid., Appendix E), while a bonds-only one is missing exactly the
    information (angles via 1,3-distances, overall shape via long-range
    distances) those targets need. `GeometricAttentionStack` itself needs no
    changes for this -- it already just runs sparse attention over whatever
    edges `graph` has.
    """
    n_graphs = int(rng.integers(n_graphs_range[0], n_graphs_range[1] + 1))
    mol_indices = rng.choice(len(dataset), size=n_graphs, replace=False)

    feature_chunks = []
    subgraphs = []
    labels = []
    for idx in mol_indices:
        d = dataset[int(idx)]
        feature_chunks.append(d.x.float())
        pos = d.pos.numpy()
        n_atoms = d.x.shape[0]
        if all_pairs:
            src, dst = np.where(~np.eye(n_atoms, dtype=bool))
        else:
            src, dst = d.edge_index.numpy()
        dist = np.linalg.norm(pos[src] - pos[dst], axis=-1)
        g = dgl.graph(
            (torch.from_numpy(src).long(), torch.from_numpy(dst).long()),
            num_nodes=n_atoms,
        )
        g.edata["distance"] = torch.from_numpy(dist).float()
        subgraphs.append(g)
        labels.append(float(d.y[0, target_idx]))

    atom_features = torch.cat(feature_chunks, dim=0)
    graph = dgl.batch(subgraphs)
    y_per_molecule = torch.tensor(labels, dtype=torch.float32)
    return atom_features, graph, y_per_molecule
