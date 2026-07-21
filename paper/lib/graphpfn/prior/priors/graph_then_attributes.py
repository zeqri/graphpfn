"""Main pipeline: graph -> attributes -> postprocessing."""

from __future__ import annotations

import torch

from lib.util import TaskType

from ..attributes import sample_attributes_gnn
from ..checks import SanityCheckError
from ..graphs import sample_graph
from ..postprocessing import (
    apply_task,
    compute_n_train_nodes,
    drop_constant_features,
    process_features,
)
from ..prior_typings import GraphThenAttributesPriorConfig, PriorDataset


def sample_dataset(config: GraphThenAttributesPriorConfig) -> PriorDataset:
    graph = sample_graph(config["graph"])
    edges = torch.stack(graph.edges(), dim=0)
    # Only geometric graph samplers (e.g. molecule-skeleton) set
    # edata["distance"]; default to zeros so PriorDataset has a
    # consistently-shaped field regardless of which sampler produced this
    # graph (needed for _pad_and_batch/DDP scatter, which treat every
    # dataset in a batch uniformly).
    edge_distance = graph.edata.get(
        "distance", torch.zeros(edges.shape[1], dtype=torch.float32)
    )
    actual_n_nodes = graph.num_nodes()

    n_train_nodes = compute_n_train_nodes(
        config["graph"]["n_nodes"], config["train_ratio"]
    )

    if n_train_nodes >= actual_n_nodes:
        raise SanityCheckError(
            f"n_train_nodes ({n_train_nodes}) >= actual_n_nodes ({actual_n_nodes}). "
            "Graph became too small after extracting largest component."
        )

    features, labels = sample_attributes_gnn(graph, config["scm"])

    features = process_features(
        features,
        p_cat=config["postprocessing"]["p_cat"],
        max_categories=config["postprocessing"]["max_categories"],
        do_permute_features=config["postprocessing"]["permute_features"],
    )

    train_features = features[:n_train_nodes, :]
    _, mask = drop_constant_features(train_features)
    features = features[:, mask]

    task_type = TaskType(config["task"]["_type_"])
    labels = apply_task(
        labels, config["task"], config["postprocessing"]["permute_labels"]
    )

    return {
        "features": features,
        "labels": labels,
        "edges": edges,
        "edge_distance": edge_distance,
        "n_train_nodes": n_train_nodes,
        "task_type": task_type,
    }
