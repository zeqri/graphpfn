"""Valence-capped molecule skeleton sampler (topology only, no distances yet).

Builds molecule-like graphs by explicitly modeling valence (not just a single
global max-degree, unlike tree_with_rings.py): sample heavy atoms (C/N/O/F)
with element-specific valence budgets, grow a tree + ring-closure skeleton
among them where each edge consumes a sampled bond order (1/2/3) worth of
valence from both endpoints (so e.g. a C#C triple bond uses 3 of carbon's 4
valence units via a single edge, leaving 1 unit -- giving that carbon degree
2 overall, as in H-C#C-H), then saturate every remaining unit of valence on
every heavy atom with an explicit hydrogen leaf.

This is the topology-only half of the chemistry-aware prior prototyped in
dev/sample_geometric_molecule_prior.py: element identity and bond order are
used here only to shape *connectivity* (which nodes end up with how many
neighbors, giving a heterogeneous, valence-consistent degree distribution
instead of one uniform cap for every node) -- they are not retained as node/
edge features, and bond *distance* is not sampled at all yet. See that dev
script's module docstring for the fuller rationale and the iteration history
that led here: sampling H as an independent node competing for attachment
slots during tree growth caused frequent valence violations (H is ~52% of
atoms, so the pool of nodes with spare valence emptied out fast); building
the heavy skeleton first and only then attaching H leaves for leftover
valence fixes this, since H can now only ever be added, never compete.
"""

from __future__ import annotations

import dgl
import numpy as np
import torch

from .tree_with_rings import _bfs_within_distance
from .util import to_simple

# Valence: total bond-order budget per atom.
VALENCE = {"H": 1, "F": 1, "O": 2, "N": 3, "C": 4}

# Composition of the *heavy* (non-H) skeleton atoms -- H is never drawn
# directly, it's an emergent consequence of leftover valence on the finished
# skeleton. Roughly matches QM9's heavy-atom mix (C dominant, O/N less
# common, F rare).
HEAVY_ELEMENTS = ["C", "N", "O", "F"]
HEAVY_ELEMENT_WEIGHTS = np.array([0.32, 0.07, 0.08, 0.01])
HEAVY_ELEMENT_WEIGHTS = HEAVY_ELEMENT_WEIGHTS / HEAVY_ELEMENT_WEIGHTS.sum()

# Bond order sampled per new edge, weighted toward single bonds, then capped
# by both endpoints' remaining valence.
BOND_ORDER_VALUES = np.array([1, 2, 3])
BOND_ORDER_WEIGHTS = np.array([0.8, 0.15, 0.05])


def _sample_bond_order(remaining_u: int, remaining_v: int) -> int:
    # max(1, ...): the phase-1 degenerate fallback below (a node with 0
    # remaining valence picked to preserve connectivity) forces order=1
    # rather than crashing in that rare case.
    cap = max(1, min(remaining_u, remaining_v, 3))
    mask = BOND_ORDER_VALUES <= cap
    weights = BOND_ORDER_WEIGHTS[mask]
    weights = weights / weights.sum()
    return int(np.random.choice(BOND_ORDER_VALUES[mask], p=weights))


def _sample_valence_capped_tree_with_rings(
    n_nodes: int,
    avg_degree: float,
    valence_cap: np.ndarray,
    *,
    min_ring_size: int,
    max_ring_size: int,
) -> dict[tuple[int, int], int]:
    """Two-phase tree + ring-closure growth (see tree_with_rings.py),
    generalized so the degree cap is replaced by a per-node *valence* budget
    that each new edge consumes by its sampled bond order (1/2/3), not just
    by 1.

    Returns {(min(u,v), max(u,v)): bond_order} for every edge.
    """
    remaining_valence = valence_cap.copy()
    adjacency: list[set[int]] = [set() for _ in range(n_nodes)]
    order_by_pair: dict[tuple[int, int], int] = {}

    def add_edge(u: int, v: int) -> None:
        order = _sample_bond_order(remaining_valence[u], remaining_valence[v])
        order_by_pair[(min(u, v), max(u, v))] = order
        adjacency[u].add(v)
        adjacency[v].add(u)
        remaining_valence[u] -= order
        remaining_valence[v] -= order

    for i in range(1, n_nodes):
        candidates = [j for j in range(i) if remaining_valence[j] >= 1]
        if not candidates:
            candidates = list(range(i))  # degenerate fallback, see docstring
        j = int(np.random.choice(candidates))
        add_edge(i, j)

    target_extra_edges = max(0, round(avg_degree * n_nodes / 2) - (n_nodes - 1))
    max_attempts = target_extra_edges * 20 + 50
    added = 0
    attempts = 0
    while added < target_extra_edges and attempts < max_attempts:
        attempts += 1
        u = int(np.random.randint(n_nodes))
        if remaining_valence[u] < 1:
            continue
        nearby = _bfs_within_distance(adjacency, u, max_ring_size - 1)
        candidates = [
            v
            for v, d in nearby.items()
            if d >= min_ring_size - 1
            and remaining_valence[v] >= 1
            and v not in adjacency[u]
        ]
        if not candidates:
            continue
        v = int(np.random.choice(candidates))
        add_edge(u, v)
        added += 1

    return order_by_pair


def sample_molecule_skeleton(
    n_nodes: int,
    avg_degree: float,
    *,
    heavy_atom_fraction: float = 0.42,
    min_ring_size: int = 3,
    max_ring_size: int = 7,
) -> dgl.DGLGraph:
    """Sample a valence-consistent molecule-like graph: a heavy-atom (C/N/O/F)
    skeleton grown with per-element valence caps and sampled bond orders, then
    saturated with explicit hydrogen leaves for every unit of leftover valence.

    Args:
        n_nodes: Target *total* node count (heavy atoms + hydrogens
            combined). Hydrogen count is emergent from leftover valence on
            the finished skeleton, not a direct count, so the actual total
            will vary somewhat around this target -- similar in spirit to how
            extract_largest_component already makes other samplers' actual
            sizes approximate rather than exact.
        avg_degree: Target average degree *of the heavy-atom skeleton only*
            (hydrogens are always degree 1 and aren't part of this budget).
            Real heavy-atom-only connectivity is much sparser than
            whole-molecule (including H) average degree -- mostly chains
            with occasional single rings, typically ~1.8-2.6 -- since most of
            the whole-molecule average degree comes from many degree-1 H's.
        heavy_atom_fraction: Fraction of n_nodes targeted as heavy atoms;
            the rest emerges as hydrogens. Calibrated empirically (see
            dev/sample_geometric_molecule_prior.py) so realized totals land
            close to n_nodes on average.
        min_ring_size: Minimum ring size (in atoms) a closing edge may
            create.
        max_ring_size: Maximum ring size a closing edge may create.

    Returns:
        Connected, simple graph (no largest-component extraction needed --
        it's connected by construction, same as tree_with_rings). Element
        identity, valence, and bond order are used only to shape this
        topology; they are not attached to the returned graph in any way.
    """
    n_heavy = max(2, round(n_nodes * heavy_atom_fraction))

    heavy_elements = np.random.choice(
        HEAVY_ELEMENTS, size=n_heavy, p=HEAVY_ELEMENT_WEIGHTS
    )
    heavy_valence_cap = np.array([VALENCE[e] for e in heavy_elements])

    order_by_pair = _sample_valence_capped_tree_with_rings(
        n_heavy,
        avg_degree,
        heavy_valence_cap,
        min_ring_size=min_ring_size,
        max_ring_size=max_ring_size,
    )

    valence_used = np.zeros(n_heavy, dtype=np.int64)
    for (u, v), order in order_by_pair.items():
        valence_used[u] += order
        valence_used[v] += order
    # clip: the phase-1 degenerate fallback can push valence_used above the
    # cap; we don't retroactively add negative hydrogens for that case.
    remaining = np.clip(heavy_valence_cap - valence_used, 0, None)

    edges = list(order_by_pair.keys())
    next_id = n_heavy
    for i in range(n_heavy):
        for _ in range(int(remaining[i])):
            edges.append((i, next_id))
            next_id += 1
    n_total = next_id

    src = torch.tensor([e[0] for e in edges] + [e[1] for e in edges], dtype=torch.int64)
    dst = torch.tensor([e[1] for e in edges] + [e[0] for e in edges], dtype=torch.int64)
    graph = dgl.graph((src, dst), num_nodes=n_total)
    return to_simple(graph)
