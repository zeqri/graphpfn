"""Main pipeline: many small graphs -> one virtual node each -> attributes.

Generates `n_graphs` small graphs per synthetic dataset, gives each one a
virtual (readout) node wired to all of its own real nodes, disjoint-unions
everything, and runs the existing GNN-SCM ONCE over the whole union so every
small graph is governed by the same underlying feature->label task
(mirroring how one graph's many nodes share one SCM in the node-level
priors). Real nodes keep their SCM-generated features but never a label;
only virtual nodes are ever context/query, matching the paper's own
node-level convention of "one context/query row = one complete labeled
example" -- just with "example" redefined as "graph" instead of "node".
"""

from __future__ import annotations

import random

import dgl
import numpy as np
import torch

from lib.util import TaskType

from ..attributes import sample_attributes_gnn
from ..checks import SanityCheckError
from ..graphs import sample_graph
from ..postprocessing import (
    compute_n_train_nodes,
    drop_constant_features,
    process_features,
    standard_scaling,
)
from ..prior_typings import GraphLevelPriorConfig, PriorDataset, SmallGraphStructureConfig

MIN_REAL_NODES = 3  # a virtual node aggregating 1-2 real nodes tests nothing


def _sample_small_graph_config(bounds: SmallGraphStructureConfig) -> dict:
    """Independently draws one small graph's structure config from `bounds`.

    Plain Python/numpy sampling rather than the generic distribution-config
    machinery: that machinery resolves a config once per whole dataset, but
    each of the `n_graphs` graphs here needs its own independent draw.
    """
    n_nodes = random.randint(bounds["min_nodes"], bounds["max_nodes"])
    avg_degree = float(np.random.uniform(bounds["avg_degree_min"], bounds["avg_degree_max"]))

    upper_groups = max(1, n_nodes // 4)  # random_partition needs min_size*n_groups <= n_nodes
    lower_groups = min(bounds["n_groups_min"], upper_groups)
    n_groups = random.randint(lower_groups, upper_groups)

    return {
        "n_nodes": n_nodes,
        "avg_degree": avg_degree,
        "sampler": {
            "_type_": "sbm",
            "n_groups": n_groups,
            "offdiagonal_coef": bounds["offdiagonal_coef"],
        },
    }


def _aggregate(values: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "mean":
        return values.mean()
    if kind == "sum":
        return values.sum()
    if kind == "max":
        return values.max()
    if kind == "min":
        return values.min()
    raise ValueError(f"Unknown label_aggregation: {kind}")


def _sample_real_graphs(config: GraphLevelPriorConfig) -> list[dgl.DGLGraph]:
    n_graphs_requested = config["n_graphs"]
    budget = config["total_n_nodes_budget"]
    bounds = config["graph"]

    # Cap per-graph size so n_graphs * (max_nodes + 1 virtual) doesn't
    # wildly exceed the total node budget (the same role `graph.n_nodes`
    # plays for the single graph in the node-level priors).
    max_nodes_by_budget = max(bounds["min_nodes"], budget // n_graphs_requested - 1)
    effective_bounds: SmallGraphStructureConfig = {
        **bounds,
        "max_nodes": min(bounds["max_nodes"], max_nodes_by_budget),
    }
    if effective_bounds["max_nodes"] < bounds["min_nodes"]:
        raise SanityCheckError(
            f"budget ({budget}) too small for n_graphs ({n_graphs_requested}) "
            f"with min_nodes={bounds['min_nodes']}"
        )

    # Retry until EXACTLY n_graphs_requested succeed (not "attempt that many
    # times and keep whatever survives"): n_graphs_requested is shared across
    # every dataset in a DDP batch (see the toml's `_shared_ = true`), and
    # downstream code uses len(real_graphs) as the actual graph count for
    # compute_n_train_nodes. If failed/too-small attempts were just dropped,
    # len(real_graphs) would vary randomly across datasets in the same
    # batch even with n_graphs_requested/train_ratio identical, breaking
    # _pad_and_batch's "all datasets share one n_train_nodes" invariant.
    real_graphs: list[dgl.DGLGraph] = []
    max_attempts = n_graphs_requested * 10 + 50
    attempts = 0
    while len(real_graphs) < n_graphs_requested and attempts < max_attempts:
        attempts += 1
        try:
            small_config = _sample_small_graph_config(effective_bounds)
            g = sample_graph(small_config)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001 -- tiny SBM draws occasionally hit degenerate cases
            continue
        if g.num_nodes() >= MIN_REAL_NODES:
            real_graphs.append(g)

    if len(real_graphs) < n_graphs_requested:
        raise SanityCheckError(
            f"Only generated {len(real_graphs)}/{n_graphs_requested} usable "
            f"small graphs within {max_attempts} attempts"
        )
    return real_graphs


def _build_disjoint_union(
    real_graphs: list[dgl.DGLGraph],
    n_context: int,
) -> tuple[dgl.DGLGraph, list[int]]:
    """Builds the [context virtual][query virtual][atoms] node layout.

    Virtual nodes must occupy a contiguous prefix -- specifically the first
    `n_context` global indices must be exactly the context virtual nodes --
    so the existing "train = [0, n_train_nodes)" convention correctly
    identifies context graphs and nothing else (real/atom nodes are never
    in that prefix, so they can never accidentally become "context").
    """
    n_graphs = len(real_graphs)
    order = list(range(n_graphs))
    random.shuffle(order)
    reordered = order[:n_context] + order[n_context:]  # context block first

    edge_src_parts: list[torch.Tensor] = []
    edge_dst_parts: list[torch.Tensor] = []
    n_real_per_graph: list[int] = []

    atom_offset = n_graphs  # atoms start right after the n_graphs virtual nodes
    for local_virtual_id, pos in enumerate(reordered):
        g = real_graphs[pos]
        n_real = g.num_nodes()
        n_real_per_graph.append(n_real)

        src, dst = g.edges()
        edge_src_parts.append(src.to(torch.int64) + atom_offset)
        edge_dst_parts.append(dst.to(torch.int64) + atom_offset)

        atoms = torch.arange(n_real, dtype=torch.int64) + atom_offset
        virtual_id = torch.full((n_real,), local_virtual_id, dtype=torch.int64)
        edge_src_parts.append(virtual_id)
        edge_dst_parts.append(atoms)
        edge_src_parts.append(atoms)
        edge_dst_parts.append(virtual_id)

        atom_offset += n_real

    n_nodes = atom_offset
    edges = torch.stack(
        [torch.cat(edge_src_parts), torch.cat(edge_dst_parts)], dim=0
    )
    graph = dgl.graph((edges[0], edges[1]), num_nodes=n_nodes)
    return graph, n_real_per_graph


def sample_dataset(config: GraphLevelPriorConfig) -> PriorDataset:
    real_graphs = _sample_real_graphs(config)
    n_graphs = len(real_graphs)

    n_context = compute_n_train_nodes(n_graphs, config["train_ratio"])
    n_context = max(1, min(n_context, n_graphs - 1))  # keep >=1 graph on each side

    big_graph, n_real_per_graph = _build_disjoint_union(real_graphs, n_context)
    n_nodes = big_graph.num_nodes()

    is_virtual = torch.zeros(n_nodes, dtype=torch.bool)
    is_virtual[:n_graphs] = True

    # >>> Attribute generation: ONE shared SCM instance for the whole
    # dataset (same "task" for every small graph, mirroring how one graph's
    # many nodes share one SCM in the node-level priors). Virtual nodes'
    # private SCM input is zeroed so they carry no signal the model can't
    # observe (their input features are zeroed below too).
    features, labels = sample_attributes_gnn(
        big_graph, config["scm"], zero_causes_mask=is_virtual
    )

    # >>> Guarantee graph-dependence: override each virtual node's label
    # with an aggregate of its own real nodes' labels. The SCM's per-channel
    # mixing mask (graph_conv_ratio) is shared across every node, so it
    # can't be biased just for virtual nodes -- at low graph_conv_ratio a
    # virtual node's own SCM-computed label would often depend on nothing
    # the model can observe (its causes are zeroed, and its features are
    # zeroed too). Aggregating its own real nodes' labels instead makes the
    # dependence deterministic rather than probabilistic.
    labels = labels.clone()
    atom_start = n_graphs
    for local_virtual_id, n_real in enumerate(n_real_per_graph):
        atom_end = atom_start + n_real
        labels[local_virtual_id] = _aggregate(
            labels[atom_start:atom_end], config["label_aggregation"]
        )
        atom_start = atom_end
    labels[n_graphs:] = 0.0  # atom labels are never read; zeroed for clarity

    # Standardize using virtual-node labels only: they're the only ones
    # that ever get used, and atoms would otherwise dilute the statistics
    # (there are usually many more atoms than virtual nodes).
    virtual_labels = labels[:n_graphs].unsqueeze(-1)
    labels[:n_graphs] = standard_scaling(virtual_labels).squeeze(-1)

    # >>> Feature postprocessing. Constant-column detection is fit on ATOM
    # rows: context virtual-node rows are all-zero placeholders and would
    # otherwise make every column look constant (same fix as the released
    # package's `feature_fit_mask` for QM9 virtual-node finetuning).
    features = process_features(
        features,
        p_cat=config["postprocessing"]["p_cat"],
        max_categories=config["postprocessing"]["max_categories"],
        do_permute_features=config["postprocessing"]["permute_features"],
    )
    atom_features = features[n_graphs:, :]
    _, col_mask = drop_constant_features(atom_features)
    features = features[:, col_mask]
    features = features.clone()
    features[is_virtual] = 0.0  # placeholder: no informative input at virtual rows

    edges = torch.stack(big_graph.edges(), dim=0)

    return {
        "features": features,
        "labels": labels,
        "edges": edges,
        "n_train_nodes": n_context,
        "task_type": TaskType.REGRESSION,
        "labeled_mask": is_virtual,
        "feature_fit_mask": ~is_virtual,
        # Labels are already standardized above (over all virtual nodes,
        # context+query) -- the eval path must not re-standardize on just
        # the (possibly single-node) train subset. See prior_typings.py.
        "labels_standardized": True,
    }
