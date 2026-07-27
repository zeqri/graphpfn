"""Graph-level label generation via a star-connected virtual readout node.

`sample_attributes_gnn`'s GNN-SCM only ever produces per-node labels. To get
one label per *molecule* (graph-level task) instead, a virtual node is
star-connected to every atom of its own molecule before running the SCM --
its post-SCM output is then a function of that molecule's atoms, propagated
through the same graph-convolution machinery (`MixedGraphLinear`/
`GeometricConv`, see layers.py) real per-atom labels already use. This is a
data-generation-time device only: it never touches the model's forward pass,
only the synthetic label generator.

Fully vectorized (no Python loop over molecules), building on the fact that
`sample_multi_graph`/`dgl.batch` never adds cross-molecule edges, so molecule
membership is recoverable from `graph.batch_num_nodes()` alone.
"""

from __future__ import annotations

import dgl
import torch

from ..prior_typings import GNNSCMConfig
from .gnn_scm import sample_attributes_gnn


def sample_graph_level_labels_via_virtual_node(
    graph: dgl.DGLGraph,
    scm_config: GNNSCMConfig,
    sentinel_distance: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (atom_features, y_per_molecule).

    atom_features: (n_atoms, n_features), in `graph`'s original node order.
    y_per_molecule: (n_graphs,), one label per sub-graph of `graph`
        (`graph.batch_num_nodes()` order).
    """
    n_atoms = graph.num_nodes()
    counts = graph.batch_num_nodes()
    n_graphs = counts.shape[0]
    atom_mol_id = torch.repeat_interleave(torch.arange(n_graphs), counts)
    virtual_ids = n_atoms + atom_mol_id

    src, dst = graph.edges()
    orig_distance = graph.edata.get(
        "distance", torch.full((graph.num_edges(),), sentinel_distance, dtype=torch.float32)
    )

    atom_idx = torch.arange(n_atoms)
    # Bidirectional star edges: atom<->its molecule's virtual node.
    star_src = torch.cat([atom_idx, virtual_ids])
    star_dst = torch.cat([virtual_ids, atom_idx])
    star_distance = torch.full((2 * n_atoms,), sentinel_distance, dtype=torch.float32)

    new_src = torch.cat([src, star_src])
    new_dst = torch.cat([dst, star_dst])
    new_distance = torch.cat([orig_distance, star_distance])

    total_nodes = n_atoms + n_graphs
    graph_aug = dgl.graph((new_src, new_dst), num_nodes=total_nodes)
    graph_aug.edata["distance"] = new_distance

    zero_cause_mask = torch.arange(total_nodes) >= n_atoms
    X, y = sample_attributes_gnn(graph_aug, scm_config, zero_cause_mask=zero_cause_mask)

    return X[:n_atoms], y[n_atoms:]
