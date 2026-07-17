"""Multi-graph sampler: combine several independent sub-graphs into one dataset.

Instead of sampling a single large graph per synthetic dataset, this sampler
draws `n_graphs` independent sub-graphs and combines them via disjoint union.
Because there are no edges between sub-graphs, the resulting adjacency is
block-diagonal: the graph adapters' 1-hop message passing / attention
naturally stays scoped to each node's own sub-graph, with no changes needed
anywhere downstream (attribute generation, postprocessing, training loop).

Sub-graph sizes can optionally be drawn with two levels of randomness so that
sub-graphs within *one* dataset stay close in size to each other (like
molecules from the same real dataset), while different datasets/training
steps can target different typical sizes:
  - `base_n_nodes`: a single "typical size" for this dataset, resolved once
    (by the normal, non-`_runtime_` config-sampling pass) before this
    function is even called.
  - each of the `n_graphs` sub-graphs then draws its actual size as
    `base_n_nodes` +/- `size_jitter` (relative), instead of each sub-graph
    picking an independent size from a wide range.
If `base_n_nodes` is omitted (the original behavior), each sub-graph instead
draws its own size independently from `sub_graph["n_nodes"]`'s own
distribution, with no size correlation across sub-graphs in the same dataset.
"""

from __future__ import annotations

import dgl
import numpy as np

from ..config import resample_config


def sample_multi_graph(
    *,
    n_graphs: int,
    sub_graph: dict,
    base_n_nodes: int | None = None,
    size_jitter: float = 0.0,
) -> dgl.DGLGraph:
    """Sample `n_graphs` independent sub-graphs and combine via disjoint union.

    Args:
        n_graphs: Number of independent sub-graphs to sample and combine.
        sub_graph: A GraphConfig-shaped template (avg_degree, sampler, and --
            only if `base_n_nodes` is None -- n_nodes) used to draw each
            sub-graph's structure. Distributions inside it are typically
            marked `_runtime_: true` so `sample_config` leaves them
            unresolved; they are then resampled fresh -- independently -- for
            each of the `n_graphs` sub-graphs via `resample_config`.
        base_n_nodes: This dataset's shared "typical" sub-graph size. Sampled
            once per dataset upstream (not per sub-graph), so all sub-graphs
            in one dataset cluster around it. If None (default, kept for
            backward compatibility with configs that don't set it), each
            sub-graph instead draws its own size independently from
            `sub_graph["n_nodes"]`'s own distribution.
        size_jitter: Relative jitter applied independently to each sub-graph's
            size around `base_n_nodes`, e.g. 0.12 means +/-12%. 0 disables
            jitter (every sub-graph gets exactly `base_n_nodes` nodes). Only
            used when `base_n_nodes` is not None.

    Returns:
        Disjoint union of `n_graphs` sub-graphs (block-diagonal adjacency,
        no edges between sub-graphs).
    """
    # Local import to avoid a circular import: graphs/__init__.py imports this
    # module, and sample_graph (which we need to recurse into) is defined
    # there. By call time, graphs/__init__.py is fully initialized.
    from . import sample_graph

    subgraphs = []
    for _ in range(n_graphs):
        subgraph_config = resample_config(sub_graph)
        if base_n_nodes is not None:
            jitter = (
                1.0 + np.random.uniform(-size_jitter, size_jitter) if size_jitter else 1.0
            )
            subgraph_config["n_nodes"] = max(int(round(base_n_nodes * jitter)), 4)
        subgraphs.append(sample_graph(subgraph_config))

    return dgl.batch(subgraphs)
