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

from functools import partial
from typing import Self, TypedDict

import delu
import dgl
import torch
from torch.utils.data import DataLoader, IterableDataset

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
        (pretrain_graph_level.py) does context/query split + standardization.

    Calls `sample_multi_graph` directly rather than going through the generic
    `sample_graph` dispatcher: the dispatcher's multi-graph branch always
    follows up with `shuffle_nodes`, which rebuilds the graph via a bare
    `dgl.graph(...)` and silently drops the `batch_num_nodes()` metadata
    `dgl.batch` set -- exactly the per-molecule membership info this whole
    pipeline depends on. Skipping it is safe here: shuffle_nodes exists so
    node-level train/test splits don't correlate with generation order, but
    this pipeline splits at molecule granularity (reordered explicitly by
    `GraphLevelGraphPFN.forward`) and processes each molecule's atoms
    permutation-invariantly (attention-based), so atom order never matters.
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


# >>> Background-prefetching sampler
#
# step_fn samples a fresh dataset synchronously on the critical path, every
# micro-step -- profiling showed this alone isn't the dominant per-step cost,
# but it's still pure Python/CPU work with zero GPU overlap, so it's worth
# recovering. Mirrors GraphPriorSampler's DataLoader-based worker mechanism
# (prior/sampler.py) exactly -- same n_workers/prefetch_factor convention --
# except workers return a plain-tensor GraphLevelSample dict rather than a
# dgl.DGLGraph: dgl graphs aren't reliably picklable across the
# multiprocessing IPC boundary (the original node-level sampler sidesteps
# this the same way, representing graphs as plain edge/edge_distance tensors
# and reconstructing the dgl.DGLGraph back in the main process). Call
# `raw_sample_to_graph` on each result to reconstruct it locally.


class GraphLevelSample(TypedDict):
    atom_features: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    distance: torch.Tensor | None
    counts: torch.Tensor
    n_atoms: int
    y_per_molecule: torch.Tensor
    train_ratio: float


def _sample_graph_level_raw(
    base_prior_config: dict, min_n_graphs: int, max_attempts: int
) -> GraphLevelSample:
    for _ in range(max_attempts):
        try:
            atom_features, graph, y_per_molecule, train_ratio = sample_graph_level_dataset(
                base_prior_config
            )
        except Exception:
            continue
        if int(graph.batch_num_nodes().shape[0]) < min_n_graphs:
            continue
        src, dst = graph.edges()
        return {
            "atom_features": atom_features,
            "src": src,
            "dst": dst,
            "distance": graph.edata.get("distance"),
            "counts": graph.batch_num_nodes(),
            "n_atoms": graph.num_nodes(),
            "y_per_molecule": y_per_molecule,
            "train_ratio": train_ratio,
        }
    raise RuntimeError(f"Could not sample a valid graph-level dataset after {max_attempts} attempts")


def raw_sample_to_graph(
    sample: GraphLevelSample,
) -> tuple[torch.Tensor, dgl.DGLGraph, torch.Tensor, float]:
    """Reconstructs the dgl.DGLGraph a worker process couldn't safely pickle.

    A bare `dgl.graph((src, dst), num_nodes=...)` reconstruction silently
    drops `batch_num_nodes()` (the same footgun `shuffle_nodes` has --
    see `sample_graph_level_dataset`'s docstring), collapsing every sample to
    a single n_graphs=1 "molecule" regardless of `sample["counts"]`.
    `set_batch_num_nodes` explicitly restores it from the counts carried
    across the worker boundary.
    """
    graph = dgl.graph((sample["src"], sample["dst"]), num_nodes=sample["n_atoms"])
    graph.set_batch_num_nodes(sample["counts"])
    if sample["distance"] is not None:
        graph.edata["distance"] = sample["distance"]
    return sample["atom_features"], graph, sample["y_per_molecule"], sample["train_ratio"]


class _GraphLevelIterableDataset(IterableDataset):
    def __init__(self, base_prior_config: dict, min_n_graphs: int = 4, max_attempts: int = 50):
        self.base_prior_config = base_prior_config
        self.min_n_graphs = min_n_graphs
        self.max_attempts = max_attempts

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> GraphLevelSample:
        return _sample_graph_level_raw(self.base_prior_config, self.min_n_graphs, self.max_attempts)


def _identity(batch: list[GraphLevelSample]) -> GraphLevelSample:
    return batch[0]


def _worker_init(worker_id: int, seed: int) -> None:
    delu.random.seed(seed + worker_id)


class GraphLevelPriorSampler:
    """Background-prefetching iterator of `GraphLevelSample`s.

    `n_workers=0` (default) samples synchronously in the calling process --
    identical behavior to calling `sample_graph_level_dataset` directly, just
    wrapped in the same interface. `n_workers>0` spawns worker subprocesses
    (via `DataLoader`) that keep sampling ahead, so the training loop mostly
    just pulls an already-ready sample instead of blocking on CPU-bound
    dataset generation.
    """

    def __init__(
        self,
        base_prior_config: dict,
        seed: int,
        n_workers: int = 0,
        prefetch_factor: int | None = None,
        min_n_graphs: int = 4,
    ) -> None:
        dataset = _GraphLevelIterableDataset(base_prior_config, min_n_graphs=min_n_graphs)
        if n_workers == 0:
            self._iterator = iter(dataset)
        else:
            loader = DataLoader(
                dataset=dataset,
                batch_size=1,
                num_workers=n_workers,
                worker_init_fn=partial(_worker_init, seed=seed),
                multiprocessing_context=torch.multiprocessing.get_context("spawn"),
                collate_fn=_identity,
                prefetch_factor=prefetch_factor,
            )
            self._iterator = iter(loader)

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> GraphLevelSample:
        return next(self._iterator)
