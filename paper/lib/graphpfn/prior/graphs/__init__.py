# Modified for MolPFN (2026) from GraphPFN.
# Changes: Integrate multi-graph, tree-with-rings and molecule-skeleton samplers.
# See paper/LICENSE and paper/NOTICE at the repository root.

"""Graph sampling building blocks."""

from ..prior_typings import GraphConfig, unpack
from .erdos_renyi import sample_erdos_renyi
from .geometric import sample_geometric
from .multi_graph import sample_multi_graph
from .molecule_skeleton import sample_molecule_skeleton
from .multi_level_sbm_with_pa import sample_multi_level_sbm_with_pa
from .preferential_attachment import sample_preferential_attachment
from .sbm import sample_sbm
from .tree_with_rings import sample_tree_with_rings
from .util import (
    extract_largest_component,
    merge_graphs,
    random_partition,
    sample_adjacency,
    shuffle_nodes,
    to_simple,
)


def sample_graph(config: GraphConfig):
    n_nodes = config["n_nodes"]
    avg_degree = config["avg_degree"]
    sampler = config["sampler"]

    if sampler["_type_"] == "multi-graph":
        # Each sub-graph is fully generated (including largest-component
        # extraction) by a nested sample_graph call inside sample_multi_graph;
        # skip the to_simple/extract_largest_component steps below since
        # extract_largest_component would otherwise collapse the disjoint
        # sub-graphs into a single component.
        #
        # NOTE: config["n_nodes"] is intentionally *not* overwritten with the
        # real combined node count here (an earlier version of this code did
        # that). graph_then_attributes.py computes n_train_nodes from
        # config["n_nodes"], and with DDP sampling (GraphPriorSamplerDDP),
        # multiple datasets sampled per step must all share the exact same
        # n_train_nodes (it's broadcast as a single scalar, never scattered
        # per-rank -- see _pad_and_batch's assertion). config["n_nodes"] is
        # marked `_shared_: true` so it's identical across every dataset in a
        # batch; overwriting it per-dataset with each one's own (independently
        # sampled) actual total broke that invariant. Instead, the "unused
        # placeholder" n_nodes/avg_degree bounds in the multi-graph configs
        # are set to approximate the real combined total (see the pretrain
        # TOML's comments), and the existing retry/sanity-check machinery
        # (n_train_nodes >= actual_n_nodes -> SanityCheckError -> retry)
        # absorbs the remaining per-dataset mismatch, same as it always did
        # for ordinary largest-component shrinkage in the single-graph case.
        graph = sample_multi_graph(**unpack(sampler))
        graph = shuffle_nodes(graph)
        return graph

    match sampler["_type_"]:
        case "sbm":
            graph = sample_sbm(
                n_nodes=n_nodes,
                avg_degree=avg_degree,
                **unpack(sampler),
            )
        case "geometric":
            graph = sample_geometric(
                n_nodes=n_nodes,
                avg_degree=avg_degree,
                **unpack(sampler),
            )
        case "preferential-attachment":
            graph = sample_preferential_attachment(
                n_nodes=n_nodes,
                avg_degree=avg_degree,
            )
        case "erdos-renyi":
            graph = sample_erdos_renyi(
                n_nodes=n_nodes,
                avg_degree=avg_degree,
            )
        case "multi-level-sbm-with-pa":
            graph = sample_multi_level_sbm_with_pa(
                n_nodes=n_nodes,
                avg_degree=avg_degree,
                **unpack(sampler),
            )
        case "tree-with-rings":
            graph = sample_tree_with_rings(
                n_nodes=n_nodes,
                avg_degree=avg_degree,
                **unpack(sampler),
            )
        case "molecule-skeleton":
            graph = sample_molecule_skeleton(
                n_nodes=n_nodes,
                avg_degree=avg_degree,
                **unpack(sampler),
            )
        case _:
            raise ValueError(f"Unknown graph sampler: {sampler['_type_']}")

    graph = to_simple(graph)
    graph, _ = extract_largest_component(graph)
    graph = shuffle_nodes(graph)

    return graph


__all__ = [
    "extract_largest_component",
    "merge_graphs",
    "random_partition",
    "sample_adjacency",
    "sample_erdos_renyi",
    "sample_geometric",
    "sample_graph",
    "sample_molecule_skeleton",
    "sample_multi_graph",
    "sample_multi_level_sbm_with_pa",
    "sample_preferential_attachment",
    "sample_sbm",
    "sample_tree_with_rings",
    "shuffle_nodes",
    "to_simple",
]
