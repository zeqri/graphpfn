from __future__ import annotations

import dgl
import numpy as np
import torch

from lib.graph.data import GraphData, GraphDataset, GraphTask, get_score
from lib.graph.util import Setting
from lib.util import PartKey, Score

from .prior_typings import PriorDataset, PriorDatasetBatch


def unbatch_prior_dataset(batch: PriorDatasetBatch) -> list[PriorDataset]:
    batch_size = batch["features"].shape[0]
    n_nodes = batch["n_nodes"]
    n_features = batch["n_features"]
    n_edges = batch["n_edges"]

    datasets: list[PriorDataset] = []
    for i in range(batch_size):
        n, f, e = n_nodes[i].item(), n_features[i].item(), n_edges[i].item()
        datasets.append(
            PriorDataset(
                features=batch["features"][i, :n, :f],
                labels=batch["labels"][i, :n],
                edges=batch["edges"][i, :, :e],
                n_train_nodes=batch["n_train_nodes"],
                task_type=batch["task_type"],
                labeled_mask=batch["labeled_mask"][i, :n],
                feature_fit_mask=batch["feature_fit_mask"][i, :n],
                labels_standardized=batch["labels_standardized"],
            )
        )
    return datasets


def convert_to_graph_dataset(
    dataset: PriorDataset,
    name: str,
    setting: Setting = Setting.TRANSDUCTIVE,
    score: Score | None = None,
) -> GraphDataset[np.ndarray]:
    n_nodes = dataset["features"].shape[0]
    n_train = dataset["n_train_nodes"]
    task_type = dataset["task_type"]
    labeled_mask = dataset["labeled_mask"].numpy()

    # >>> Build masks. "test" is additionally restricted to labeled_mask so
    # that non-labeled nodes (e.g. atoms in the graph_level prior) are never
    # scored as if they were real query examples -- for the node-level
    # priors labeled_mask is all-True, so this is a no-op there.
    node_idx = np.arange(n_nodes)
    masks: dict[PartKey, np.ndarray] = {
        "train": node_idx < n_train,
        "val": np.zeros(n_nodes, dtype=bool),
        "test": (node_idx >= n_train) & labeled_mask,
    }

    # >>> Build graph
    edges = dataset["edges"]
    graph = dgl.graph(
        data=(edges[0], edges[1]),
        num_nodes=n_nodes,
        idtype=torch.int32,
    )

    labels = dataset["labels"].numpy().astype(np.float32)
    features = dataset["features"].numpy().astype(np.float32)
    feature_fit_mask = dataset["feature_fit_mask"].numpy()

    data = GraphData(
        name=name,
        graph=graph,
        labels=labels,
        masks=masks,
        num_features=features,
        cat_features=None,
        frac_features=None,
        feature_fit_mask=feature_fit_mask,
        labels_standardized=dataset["labels_standardized"],
    )
    task = GraphTask(
        labels=labels,
        masks=masks,
        type_=task_type,
        setting=setting,
        score=score if score is not None else get_score(task_type),
    )
    return GraphDataset(data, task)
