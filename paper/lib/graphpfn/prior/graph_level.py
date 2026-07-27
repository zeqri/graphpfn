"""Graph-level ("whole-molecule") prior dataset sampling.

Thin composition helper mirroring `priors/graph_then_attributes.py`'s
`sample_dataset`, but producing one label per molecule (via
`sample_graph_level_labels_via_virtual_node`) instead of one per node, and
splitting train/query at molecule granularity instead of node granularity.
This is a parallel vertical, not a modification of the node-level prior
pipeline -- see the plan for why (TaskType/GraphTask/evaluate_dataset all
assume node-level transductive semantics).
"""

from __future__ import annotations

import dgl
import torch

from .attributes import sample_graph_level_labels_via_virtual_node
from .config import sample_configs
from .graphs.multi_graph import sample_multi_graph
from .postprocessing import process_features
from .prior_typings import unpack


def sample_graph_level_dataset(
    base_prior_config: dict,
) -> tuple[torch.Tensor, dgl.DGLGraph, torch.Tensor, float]:
    """Returns (atom_features, graph, y_per_molecule, train_ratio).

    atom_features: (n_atoms, n_features).
    graph: block-diagonal multi-molecule dgl.DGLGraph (edata["distance"] set
        when the underlying graph sampler is geometric, e.g. molecule-skeleton).
    y_per_molecule: (n_graphs,) float labels, one per sub-graph of `graph`
        (`graph.batch_num_nodes()` order) -- not yet standardized; the caller
        does context/query split + standardization.

    Calls `sample_multi_graph` directly rather than going through the generic
    `sample_graph` dispatcher: the dispatcher's multi-graph branch always
    follows up with `shuffle_nodes`, which rebuilds the graph via a bare
    `dgl.graph(...)` and silently drops the `batch_num_nodes()` metadata
    `dgl.batch` set -- exactly the per-molecule membership info this whole
    pipeline depends on. Skipping it is safe here: shuffle_nodes exists so
    node-level train/test splits don't correlate with generation order, but
    this pipeline splits at molecule granularity and processes each
    molecule's atoms permutation-invariantly, so atom order never matters.
    """
    configs = sample_configs(base_prior_config, batch_size=1)
    config = configs[0]["prior"]

    graph = sample_multi_graph(**unpack(config["graph"]["sampler"]))
    atom_features, y_per_molecule = sample_graph_level_labels_via_virtual_node(
        graph, config["scm"]
    )

    atom_features = process_features(
        atom_features,
        p_cat=config["postprocessing"]["p_cat"],
        max_categories=config["postprocessing"]["max_categories"],
        do_permute_features=config["postprocessing"]["permute_features"],
    )

    return atom_features, graph, y_per_molecule, config["train_ratio"]
