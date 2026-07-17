"""Tree + ring-closure sampler: a molecule-like alternative to SBM/PA.

Unlike the SBM/preferential-attachment family (built to resemble social
networks: dense community blocks, power-law "hub" degree distributions,
only *probabilistically* connected), organic molecules are:
  - connected by construction -- every atom is transitively bonded to the
    rest, always;
  - close to trees -- a molecule with N heavy atoms and R rings has exactly
    N - 1 + R edges, and R is typically small (0-3) for small/drug-like
    molecules, so average degree sits around ~2-2.4;
  - degree-bounded by valence (mostly 1-4), not heavy-tailed.

This sampler builds a random tree (guaranteeing connectivity with no
extract_largest_component needed at all), growing it with an explicit
max-degree cap so no hub nodes emerge, then closes a small number of extra
"ring" edges between nodes that are already close in tree-distance (so ring
sizes stay realistic, e.g. 3-7 atoms), again respecting the degree cap.
"""

from __future__ import annotations

import dgl
import numpy as np
import torch

from .util import to_simple


def _bfs_within_distance(
    adjacency: list[set[int]], start: int, max_dist: int
) -> dict[int, int]:
    """Return {node: distance} for nodes within max_dist hops of start
    (excluding start itself)."""
    dist = {start: 0}
    frontier = [start]
    d = 0
    while frontier and d < max_dist:
        d += 1
        next_frontier = []
        for u in frontier:
            for v in adjacency[u]:
                if v not in dist:
                    dist[v] = d
                    next_frontier.append(v)
        frontier = next_frontier
    del dist[start]
    return dist


def sample_tree_with_rings(
    n_nodes: int,
    avg_degree: float,
    *,
    max_degree: int = 4,
    min_ring_size: int = 3,
    max_ring_size: int = 7,
) -> dgl.DGLGraph:
    """Sample a degree-capped random tree with a small number of ring-closing
    edges, resembling organic molecule topology.

    Args:
        n_nodes: Number of nodes.
        avg_degree: Target average degree. Since a tree alone has average
            degree ~2(N-1)/N =~ 2, extra "ring-closing" edges are added to
            reach this target: n_extra = max(0, round(avg_degree*N/2) -
            (N-1)). Values well above ~2.5-3 will need many rings and may not
            be achievable while respecting max_degree/min_ring_size within
            the retry budget -- the sampler simply adds as many valid edges
            as it can find rather than erroring.
        max_degree: Maximum degree any node may reach, both during tree
            growth and ring-closure (mimics valence: e.g. 4 for carbon-like
            atoms).
        min_ring_size: Minimum ring size (in atoms) a closing edge may
            create. A closing edge between two nodes at tree-distance d
            creates a ring of size d + 1.
        max_ring_size: Maximum ring size a closing edge may create.

    Returns:
        Connected, simple, degree-capped graph (no largest-component
        extraction needed -- it's connected by construction).
    """
    assert n_nodes >= 1
    assert max_degree >= 2
    assert min_ring_size >= 3
    assert max_ring_size >= min_ring_size

    degree = [0] * n_nodes
    adjacency: list[set[int]] = [set() for _ in range(n_nodes)]
    edges: list[tuple[int, int]] = []

    def add_edge(u: int, v: int) -> None:
        edges.append((u, v))
        adjacency[u].add(v)
        adjacency[v].add(u)
        degree[u] += 1
        degree[v] += 1

    # >>> Random degree-capped tree growth: each new node attaches to a
    # uniformly random *existing* node that still has spare degree capacity.
    # This keeps the tree connected by construction while avoiding the
    # "richer get richer" hub bias of plain random recursive trees / PA.
    for i in range(1, n_nodes):
        candidates = [j for j in range(i) if degree[j] < max_degree]
        if not candidates:
            candidates = list(range(i))  # degenerate fallback, shouldn't
            # normally trigger since max_degree >= 2 leaves room as the tree
            # grows, but guards against pathological small max_degree.
        j = int(np.random.choice(candidates))
        add_edge(i, j)

    # >>> Ring closure: add a small number of extra edges between nodes
    # already close in tree-distance, so ring sizes stay realistic.
    target_extra_edges = max(0, round(avg_degree * n_nodes / 2) - (n_nodes - 1))
    max_attempts = target_extra_edges * 20 + 50
    added = 0
    attempts = 0
    while added < target_extra_edges and attempts < max_attempts:
        attempts += 1
        u = int(np.random.randint(n_nodes))
        if degree[u] >= max_degree:
            continue
        nearby = _bfs_within_distance(adjacency, u, max_ring_size - 1)
        candidates = [
            v
            for v, d in nearby.items()
            if d >= min_ring_size - 1 and degree[v] < max_degree and v not in adjacency[u]
        ]
        if not candidates:
            continue
        v = int(np.random.choice(candidates))
        add_edge(u, v)
        added += 1

    src = torch.tensor([e[0] for e in edges] + [e[1] for e in edges], dtype=torch.int64)
    dst = torch.tensor([e[1] for e in edges] + [e[0] for e in edges], dtype=torch.int64)
    graph = dgl.graph((src, dst), num_nodes=n_nodes)
    return to_simple(graph)
