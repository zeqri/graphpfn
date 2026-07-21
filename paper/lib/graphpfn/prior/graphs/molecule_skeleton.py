"""Valence-capped molecule skeleton sampler, now with QM9-derived bond
distances.

Builds molecule-like graphs by explicitly modeling valence (not just a single
global max-degree, unlike tree_with_rings.py): sample heavy atoms (C/N/O/F)
with element-specific valence budgets, grow a tree + ring-closure skeleton
among them where each edge consumes a sampled bond order (1/2/3) worth of
valence from both endpoints (so e.g. a C#C triple bond uses 3 of carbon's 4
valence units via a single edge, leaving 1 unit -- giving that carbon degree
2 overall, as in H-C#C-H), then saturate every remaining unit of valence on
every heavy atom with an explicit hydrogen leaf. Every edge is then annotated
with a bond distance sampled from a per-element-pair QM9 fit, exposed as
`graph.edata["distance"]`.

Element identity and bond order are used here only to shape *connectivity*
(which nodes end up with how many neighbors, giving a heterogeneous,
valence-consistent degree distribution instead of one uniform cap for every
node) and to look up the per-edge distance prior -- elements/bond order
themselves are not retained as node/edge features. By default, distance does
NOT depend on the sampled bond order (e.g. C=C vs C-C draw from the same
pooled C-C distance prior, BOND_PRIOR) -- see dev/sample_geometric_molecule_
prior.py, the standalone prototype this was ported from, for the fuller
rationale and iteration history: sampling H as an independent node competing
for attachment slots during tree growth caused frequent valence violations
(H is ~52% of atoms, so the pool of nodes with spare valence emptied out
fast); building the heavy skeleton first and only then attaching H leaves
for leftover valence fixes this, since H can now only ever be added, never
compete.

An opt-in `bond_order_aware_distances` flag (default False, so every
existing config's behavior is unchanged) switches to BOND_PRIOR_BY_ORDER
instead, a per-(element-pair, bond-order) fit -- real QM9 bond distances
measured directly (not textbook estimates) show this pooling hides
substantial variation for pairs where both elements have valence >= 2 (e.g.
C-O: pooled std 0.0905 A, vs. 0.0334/0.0090 A once split into single/double)
-- see BOND_PRIOR_BY_ORDER's own comment for the measured numbers.
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

# >>> QM9-derived bond-length priors: {frozenset({elem_i, elem_j}): (mean_A,
# std_A)}, ported from dev/sample_geometric_molecule_prior.py. Bond order
# (single/double/triple) is not distinguished yet -- see module docstring.
BOND_PRIOR: dict[frozenset, tuple[float, float]] = {
    frozenset({"C", "C"}): (1.5042, 0.0719),
    frozenset({"C", "F"}): (1.3332, 0.0115),
    frozenset({"C", "H"}): (1.0929, 0.0068),
    frozenset({"C", "N"}): (1.3918, 0.1011),
    frozenset({"C", "O"}): (1.3689, 0.0906),
    frozenset({"H", "N"}): (1.0128, 0.0059),
    frozenset({"H", "O"}): (0.9645, 0.0030),
    frozenset({"N", "N"}): (1.3311, 0.0382),
    frozenset({"N", "O"}): (1.3982, 0.0503),
}
# Fallback for element pairs not covered above (e.g. H-H, H-F, F-F, F-N, F-O,
# O-O): a generic single-bond-ish distance with a wide std, since it's a
# placeholder rather than a QM9 fit.
DEFAULT_BOND_PRIOR = (1.45, 0.15)

# >>> QM9-derived bond-length priors STRATIFIED BY BOND ORDER:
# {(frozenset({elem_i, elem_j}), bond_order): (mean_A, std_A)}. Measured
# directly from all 130,831 QM9 molecules (real mol.pos distances grouped by
# RDKit bond type), not textbook values -- see dev/README.md section 13.
# Only covers pairs where BOTH elements have valence >= 2 (C, N, O), since
# H/F have valence 1 and can therefore only ever form single bonds -- for
# any pair involving H or F, BOND_PRIOR's existing pooled entry is already
# equivalent to a pure single-order fit (nothing to split), so those are
# deliberately left out here and handled by the fallback in
# _sample_bond_distances. Combinations not measured (rare in QM9, e.g. a
# sampled N-N or N-O triple bond) also fall back to the pooled BOND_PRIOR.
BOND_PRIOR_BY_ORDER: dict[tuple[frozenset, int], tuple[float, float]] = {
    (frozenset({"C", "C"}), 1): (1.5214, 0.0385),
    (frozenset({"C", "C"}), 2): (1.3624, 0.0309),
    (frozenset({"C", "C"}), 3): (1.2039, 0.0063),
    (frozenset({"C", "N"}), 1): (1.4254, 0.0721),
    (frozenset({"C", "N"}), 2): (1.3117, 0.0305),
    (frozenset({"C", "N"}), 3): (1.1563, 0.0016),
    (frozenset({"C", "O"}), 1): (1.4135, 0.0334),
    (frozenset({"C", "O"}), 2): (1.2055, 0.0090),
    (frozenset({"N", "N"}), 1): (1.3479, 0.0224),
    (frozenset({"N", "N"}), 2): (1.2897, 0.0359),
    (frozenset({"N", "O"}), 1): (1.4069, 0.0382),
    (frozenset({"N", "O"}), 2): (1.2280, 0.0084),
}


def _sample_bond_distances(
    src: torch.Tensor,
    dst: torch.Tensor,
    elements: np.ndarray,
    bond_order_by_pair: dict[tuple[int, int], int] | None = None,
) -> torch.Tensor:
    """Sample a bond distance per edge from the per-element-pair QM9 prior.

    Distances are sampled once per undirected edge (not once per directed
    edge) so both directions of a bond carry the same physical distance --
    `src`/`dst` here are already the doubled (both-directions) edge lists
    matching graph.edges()'s order, same convention as the rest of this
    module (see sample_molecule_skeleton's own src/dst construction).

    Args:
        bond_order_by_pair: If given (keyed by (min(u,v), max(u,v))), looks
            up BOND_PRIOR_BY_ORDER[(element_pair, order)] first, falling
            back to the pooled BOND_PRIOR/DEFAULT_BOND_PRIOR for any
            (pair, order) combination it doesn't cover. If None (default),
            uses the original order-agnostic BOND_PRIOR lookup unchanged --
            this is the behavior every pre-existing config still gets.
    """
    src_l, dst_l = src.tolist(), dst.tolist()
    undirected_distance: dict[tuple[int, int], float] = {}
    for u, v in zip(src_l, dst_l):
        key = (min(u, v), max(u, v))
        if key in undirected_distance:
            continue
        pair = frozenset({elements[u], elements[v]})
        mu_sigma = None
        if bond_order_by_pair is not None:
            order = bond_order_by_pair.get(key, 1)
            mu_sigma = BOND_PRIOR_BY_ORDER.get((pair, order))
        if mu_sigma is None:
            mu_sigma = BOND_PRIOR.get(pair, DEFAULT_BOND_PRIOR)
        mu, sigma = mu_sigma
        undirected_distance[key] = max(0.05, float(np.random.normal(mu, sigma)))

    distances = [undirected_distance[(min(u, v), max(u, v))] for u, v in zip(src_l, dst_l)]
    return torch.tensor(distances, dtype=torch.float32)


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
    compute_distances: bool = True,
    bond_order_aware_distances: bool = False,
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
        compute_distances: If True (default), sample and attach a per-edge
            QM9-derived bond distance (see BOND_PRIOR) as
            `graph.edata["distance"]`. Set to False for experiments that
            deliberately want a topology-only ablation -- e.g. testing
            whether widening avg_degree alone changes downstream behavior,
            without also exercising model.py's distance-bias attention
            pathway (which activates whenever "distance" is present in
            edata, regardless of whether an experiment's SCM ever samples
            the "geometric-rbf" conv_type). Without this flag, EVERY
            molecule-skeleton graph would carry real distances unconditionally,
            silently contaminating any "no geometry" ablation.
        bond_order_aware_distances: If True, look up distances from
            BOND_PRIOR_BY_ORDER (keyed by element pair AND the edge's
            sampled bond order) instead of the pooled, order-agnostic
            BOND_PRIOR. Defaults to False so every pre-existing config's
            distance distribution is completely unchanged -- only takes
            effect when both this and compute_distances are True, and only
            for element pairs BOND_PRIOR_BY_ORDER actually covers (anything
            else falls back to the pooled prior, same as before).

    Returns:
        Connected, simple graph (no largest-component extraction needed --
        it's connected by construction, same as tree_with_rings). If
        `compute_distances` is True, `graph.edata["distance"]` is set to a
        per-edge bond distance sampled from the QM9-derived BOND_PRIOR for
        that edge's element pair; otherwise the graph has no edata at all.
        Element identity, valence, and bond order themselves are only ever
        used to shape topology and (if enabled) look up the distance prior;
        they are not otherwise attached to the returned graph.
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
    graph = to_simple(graph)

    if compute_distances:
        # Elements aligned with final node ids: heavy atoms keep their
        # sampled element, every appended H leaf is "H" -- needed only to
        # look up each edge's distance prior, not retained as a node
        # feature. Distances are sampled from graph.edges() AFTER to_simple
        # (which may reorder/dedupe edges), not from the src/dst built
        # above, so that graph.edata["distance"] stays aligned with the
        # graph actually returned.
        elements = np.concatenate([heavy_elements, np.full(n_total - n_heavy, "H")])
        bond_order_by_pair = None
        if bond_order_aware_distances:
            # order_by_pair already covers every heavy-heavy edge; every
            # heavy-H edge (appended after order_by_pair's own keys in
            # `edges`) is always order 1 -- H's valence of 1 makes any
            # other order impossible, so this is exact, not an
            # approximation.
            bond_order_by_pair = dict(order_by_pair)
            for u, v in edges[len(order_by_pair):]:
                bond_order_by_pair[(min(u, v), max(u, v))] = 1
        final_src, final_dst = graph.edges()
        graph.edata["distance"] = _sample_bond_distances(
            final_src, final_dst, elements, bond_order_by_pair
        )
    return graph
