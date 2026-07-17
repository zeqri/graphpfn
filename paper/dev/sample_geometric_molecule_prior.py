"""Prototype: layer valence/bond-order-aware atom types + QM9-derived bond
distances on top of the tree-with-rings molecule sampler, and visualize the
result.

Standalone exploration script -- does NOT touch the training pipeline
(sample_graph, PriorDataset, GraphPFN.forward, the graph adapters, ...).

Why this reimplements tree-with-rings' two-phase algorithm locally rather
than calling sample_graph/sample_tree_with_rings: the original algorithm
caps *degree* by one global scalar (max_degree=4), independent of what atom
ends up on each node.

Two iterations led here:
  - v1 sampled an element per node independently of topology, then annotated
    edges with QM9 distances. Chemically inconsistent (a degree-4 nitrogen
    with 3 H already attached could still pick up a 4th bond).
  - v2 sampled elements first, derived a per-node *valence* budget (H/F=1,
    O=2, N=3, C=4), and grew the tree/rings with each new edge consuming a
    sampled *bond order* (1/2/3) worth of valence from both endpoints
    instead of just 1 -- this is what makes valence, not degree, the capped
    quantity: a C#C triple bond consumes 3 of carbon's 4 valence units via a
    single edge (degree contribution 1), leaving 1 unit for one more bond,
    giving that carbon degree 2 overall (H-C#C-H). But since H/F nodes were
    drawn independently and still had to compete uniformly at random for an
    attachment slot during tree growth, and H alone is ~52% of atoms, the
    pool of nodes with spare valence emptied out fast as the tree grew,
    forcing a "degenerate" fallback (attach anyway, exceeding valence) on
    ~10% of attachments.
  - v3 (current): removes that competition entirely by not sampling H as an
    independent node at all. Instead: sample only the heavy atoms (C/N/O/F)
    and grow the valence-capped tree/rings *among those*, then walk the
    finished skeleton and attach one explicit H leaf per unit of *leftover*
    valence on each heavy atom. H can now only ever be added, never compete
    for a slot, so valence violations can no longer happen for H, and the
    heavy-only valence distribution (mean ~3.3, vs. ~1.8 once H is mixed in)
    makes the skeleton-growth fallback far rarer too. See
    sample_skeleton_then_hydrogens.

Bond *distance* is sampled per edge from a per-element-pair Gaussian fit to
QM9 (mean, std), with a generic fallback for element pairs QM9 doesn't
cover. Distance does NOT yet depend on the sampled bond order (e.g. C=C vs
C-C get the same C-C distance prior for now) -- deferred deliberately, bond
order is being introduced here only to fix the valence/degree bookkeeping.

Three outputs:
  1. A gallery of individual molecule-like sub-graphs: node color = element,
     edge color = sampled distance, edge width = sampled bond order --
     eyeball plausibility check.
  2. A distance-validation panel: aggregate sampled distances per element
     pair across many sub-graphs, overlaid on the target QM9 N(mu, sigma).
  3. A valence-validation panel: realized degree and realized valence usage
     per element, aggregated across many sub-graphs, checked against each
     element's valence cap.

Usage:
    uv run dev/sample_geometric_molecule_prior.py --n-gallery 6 --n-validation 300
"""

import argparse
import random
import sys
import tomllib
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent.parent
if str(PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_DIR))

import dgl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from lib.graphpfn.prior.config import resample_config, sample_configs  # noqa: E402
from lib.graphpfn.prior.graphs import to_simple  # noqa: E402

CONFIG_PATH = PAPER_DIR / "exp/graphpfn/pretrain/multigraph_molecule_tree/pretrain.toml"

# >>> QM9-derived bond-length priors: {frozenset({elem_i, elem_j}): (mean_A, std_A)}.
# One entry per unordered element pair -- bond order (single/double/triple)
# is not distinguished yet (see module docstring).
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

# All elements that can appear on a node (used for colors/distance lookup).
ELEMENTS = ["H", "C", "N", "O", "F"]

# Composition of the *heavy* (non-H) skeleton atoms -- H is never drawn
# directly, see sample_skeleton_then_hydrogens. Roughly matches QM9's
# heavy-atom mix (C dominant, O/N less common, F rare).
HEAVY_ELEMENTS = ["C", "N", "O", "F"]
HEAVY_ELEMENT_WEIGHTS = np.array([0.32, 0.07, 0.08, 0.01])
HEAVY_ELEMENT_WEIGHTS = HEAVY_ELEMENT_WEIGHTS / HEAVY_ELEMENT_WEIGHTS.sum()

# Target fraction of the final atom count spent on heavy atoms; H count is
# an *emergent* consequence of leftover valence on the finished skeleton
# (see sample_skeleton_then_hydrogens), not an independent draw, so this is
# only a dial for landing near the configured n_nodes target on average --
# calibrated empirically by running this script and comparing emergent vs.
# target size (n=X vs base_n_nodes in the gallery/valence panel titles).
HEAVY_ATOM_FRACTION = 0.42

# The sub_graph template's avg_degree (3.0-6.0, from
# multigraph_molecule_tree/pretrain.toml) was calibrated for the *old*
# whole-molecule semantics, where most nodes were degree-1 H's dragging the
# average down -- reusing it directly on the heavy-only skeleton asks for
# 3-6 heavy-heavy bonds per heavy atom, far denser than real small organic
# molecules (typical heavy-atom-only average degree ~2.0-2.6: mostly chains
# with occasional single rings, not fused polycyclic meshes). That density
# consumes nearly all valence in heavy-heavy bonds, starving H generation
# (see plot_gallery/plot_valence_validation before this range was added).
# Sampled independently per sub-graph, log-uniform.
HEAVY_AVG_DEGREE_MIN = 1.8
HEAVY_AVG_DEGREE_MAX = 2.6

ELEMENT_COLORS = {
    "H": "#dddddd",
    "C": "#333333",
    "N": "#3060f0",
    "O": "#e03030",
    "F": "#30c030",
}

# >>> Valence: total bond-order budget per atom. Distinct from *degree*
# (number of neighbors) -- see module docstring's C#C example.
VALENCE = {"H": 1, "F": 1, "O": 2, "N": 3, "C": 4}

# Bond order sampled per new edge, weighted toward single bonds, then capped
# by both endpoints' remaining valence.
BOND_ORDER_VALUES = np.array([1, 2, 3])
BOND_ORDER_WEIGHTS = np.array([0.8, 0.15, 0.05])


def sample_heavy_avg_degree() -> float:
    """Log-uniform in [HEAVY_AVG_DEGREE_MIN, HEAVY_AVG_DEGREE_MAX]."""
    return float(
        np.exp(
            np.random.uniform(
                np.log(HEAVY_AVG_DEGREE_MIN), np.log(HEAVY_AVG_DEGREE_MAX)
            )
        )
    )


def load_prior_distribution_config() -> dict:
    with open(CONFIG_PATH, "rb") as f:
        data = tomllib.load(f)
    return data["base_config"]["prior"]


def _bfs_within_distance(
    adjacency: list[set[int]], start: int, max_dist: int
) -> dict[int, int]:
    """Copied from lib.graphpfn.prior.graphs.tree_with_rings (a private
    helper there, not part of its public API) -- needed here because the
    valence/bond-order-capped variant below reimplements the whole two-phase
    algorithm rather than calling sample_tree_with_rings directly."""
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


def sample_bond_order(remaining_u: int, remaining_v: int) -> int:
    # max(1, ...): the phase-1 degenerate fallback (see
    # sample_valence_capped_tree_with_rings) may pick a node with 0
    # remaining valence to preserve connectivity; force order=1 rather than
    # crash in that rare case -- it shows up as an over-cap count in
    # plot_valence_validation instead.
    cap = max(1, min(remaining_u, remaining_v, 3))
    mask = BOND_ORDER_VALUES <= cap
    weights = BOND_ORDER_WEIGHTS[mask]
    weights = weights / weights.sum()
    return int(np.random.choice(BOND_ORDER_VALUES[mask], p=weights))


def sample_valence_capped_tree_with_rings(
    n_nodes: int,
    avg_degree: float,
    valence_cap: np.ndarray,
    *,
    min_ring_size: int = 3,
    max_ring_size: int = 7,
) -> tuple[dgl.DGLGraph, dict[tuple[int, int], int], int]:
    """Two-phase tree + ring-closure growth (see
    lib.graphpfn.prior.graphs.tree_with_rings), generalized so the degree cap
    is replaced by a per-node *valence* budget (valence_cap[i]) that each new
    edge consumes by its sampled bond order (1/2/3), not just by 1.

    Returns:
        graph: connected, simple, valence-respecting molecule graph.
        order_by_pair: {(min(u,v), max(u,v)): bond_order} for every edge,
            keyed by undirected node pair -- safe to look up against
            graph.edges() after to_simple, since to_simple only reorders/
            deduplicates edge storage for an input that's already simple; it
            never renumbers nodes or changes which pairs are connected.
        n_degenerate: number of phase-1 attachments that had to ignore the
            valence cap because every earlier node was already saturated
            (should be ~0; large values mean the sampled element mix is too
            valence-starved for this n_nodes -- e.g. too many H's).
    """
    assert n_nodes >= 1
    assert min_ring_size >= 3
    assert max_ring_size >= min_ring_size

    remaining_valence = valence_cap.copy()
    adjacency: list[set[int]] = [set() for _ in range(n_nodes)]
    order_by_pair: dict[tuple[int, int], int] = {}
    n_degenerate = 0

    def add_edge(u: int, v: int) -> None:
        order = sample_bond_order(remaining_valence[u], remaining_valence[v])
        order_by_pair[(min(u, v), max(u, v))] = order
        adjacency[u].add(v)
        adjacency[v].add(u)
        remaining_valence[u] -= order
        remaining_valence[v] -= order

    for i in range(1, n_nodes):
        candidates = [j for j in range(i) if remaining_valence[j] >= 1]
        if not candidates:
            candidates = list(range(i))  # degenerate fallback, see docstring
            n_degenerate += 1
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

    edges = list(order_by_pair.keys())
    src = torch.tensor([e[0] for e in edges] + [e[1] for e in edges], dtype=torch.int64)
    dst = torch.tensor([e[1] for e in edges] + [e[0] for e in edges], dtype=torch.int64)
    graph = dgl.graph((src, dst), num_nodes=n_nodes)
    return to_simple(graph), order_by_pair, n_degenerate


def sample_skeleton_then_hydrogens(
    n_heavy: int,
    avg_degree: float,
    *,
    min_ring_size: int = 3,
    max_ring_size: int = 7,
) -> tuple[dgl.DGLGraph, np.ndarray, dict[tuple[int, int], int], int]:
    """Build the heavy-atom (C/N/O/F) skeleton first via
    sample_valence_capped_tree_with_rings, then saturate every remaining
    unit of valence on every heavy atom with an explicit hydrogen leaf.

    Unlike drawing all elements (including H) up front and growing them
    together, H never competes for an attachment slot here -- it can only
    ever be *added* after the skeleton is finished, so valence violations
    from running out of nodes with spare capacity can no longer happen for
    H, and are far rarer for the heavy skeleton too (its valence
    distribution, mean ~3.3, is much less skewed than the full mix with H,
    mean ~1.8).

    Returns:
        graph: full graph (heavy skeleton + explicit H leaves). Node ids
            0..n_heavy-1 are the heavy atoms in their original order;
            n_heavy..n_total-1 are the appended H leaves.
        elements: (n_total,) elements aligned with graph node ids.
        order_by_pair: bond order per undirected edge (min(u,v), max(u,v))
            -> order; H-heavy bonds are always order 1.
        n_degenerate: degenerate attachments during heavy-skeleton growth
            only (see sample_valence_capped_tree_with_rings) -- should be
            near 0.
    """
    heavy_elements = np.random.choice(
        HEAVY_ELEMENTS, size=n_heavy, p=HEAVY_ELEMENT_WEIGHTS
    )
    heavy_valence_cap = np.array([VALENCE[e] for e in heavy_elements])

    skeleton, order_by_pair, n_degenerate = sample_valence_capped_tree_with_rings(
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
    # clip: the degenerate fallback can push valence_used above the cap; we
    # don't retroactively add negative hydrogens for that, we just add none.
    remaining = np.clip(heavy_valence_cap - valence_used, 0, None)

    edges = list(order_by_pair.items())  # [((u, v), order), ...]
    next_id = n_heavy
    for i in range(n_heavy):
        for _ in range(int(remaining[i])):
            edges.append(((i, next_id), 1))
            next_id += 1
    n_total = next_id

    elements = np.concatenate([heavy_elements, np.full(n_total - n_heavy, "H")])

    src = torch.tensor(
        [e[0][0] for e in edges] + [e[0][1] for e in edges], dtype=torch.int64
    )
    dst = torch.tensor(
        [e[0][1] for e in edges] + [e[0][0] for e in edges], dtype=torch.int64
    )
    graph = to_simple(dgl.graph((src, dst), num_nodes=n_total))
    order_by_pair_full = {pair: order for pair, order in edges}
    return graph, elements, order_by_pair_full, n_degenerate


def sample_bond_distances(
    graph: dgl.DGLGraph, elements: np.ndarray
) -> tuple[np.ndarray, list[bool]]:
    """Sample a bond distance per edge from the per-element-pair QM9 prior.

    Distances are sampled once per undirected edge (not once per directed
    edge) so both directions of a bond carry the same physical distance.
    """
    src, dst = graph.edges()
    src_l, dst_l = src.tolist(), dst.tolist()

    undirected_distance: dict[tuple[int, int], float] = {}
    undirected_fallback: dict[tuple[int, int], bool] = {}
    for u, v in zip(src_l, dst_l):
        key = (min(u, v), max(u, v))
        if key in undirected_distance:
            continue
        pair = frozenset({elements[u], elements[v]})
        mu, sigma = BOND_PRIOR.get(pair, DEFAULT_BOND_PRIOR)
        undirected_distance[key] = max(0.05, float(np.random.normal(mu, sigma)))
        undirected_fallback[key] = pair not in BOND_PRIOR

    distances = np.array(
        [undirected_distance[(min(u, v), max(u, v))] for u, v in zip(src_l, dst_l)]
    )
    used_fallback = [
        undirected_fallback[(min(u, v), max(u, v))] for u, v in zip(src_l, dst_l)
    ]
    return distances, used_fallback


def sample_subgraph_with_chemistry(dist_config: dict):
    """Sample one molecule end to end: heavy-atom skeleton -> hydrogen
    saturation -> per-edge distances. Reuses the n_nodes/avg_degree/
    ring-size distribution from multigraph_molecule_tree/pretrain.toml's
    sub_graph template (its max_degree is ignored -- superseded by per-node
    valence caps); n_nodes is treated as a *target* for heavy+H combined
    (see HEAVY_ATOM_FRACTION), not a hard count, since final H count is
    emergent from leftover valence on the skeleton.
    """
    sampled_config = sample_configs(dist_config, batch_size=1)[0]
    graph_config = sampled_config["prior"]["graph"]
    assert graph_config["sampler"]["_type_"] == "multi-graph", (
        "Expected exp/graphpfn/pretrain/multigraph_molecule_tree/pretrain.toml"
    )
    sampler_config = graph_config["sampler"]
    sub_graph_template = sampler_config["sub_graph"]
    base_n_nodes = sampler_config.get("base_n_nodes")
    size_jitter = sampler_config.get("size_jitter", 0.0)

    subgraph_config = resample_config(sub_graph_template)
    if base_n_nodes is not None:
        jitter = 1.0 + np.random.uniform(-size_jitter, size_jitter) if size_jitter else 1.0
        target_n_nodes = max(int(round(base_n_nodes * jitter)), 4)
    else:
        target_n_nodes = subgraph_config["n_nodes"]
    # NOTE: subgraph_config["avg_degree"] (3.0-6.0) is intentionally not used
    # here -- it's calibrated for the old whole-molecule semantics. See
    # HEAVY_AVG_DEGREE_MIN/MAX.
    avg_degree = sample_heavy_avg_degree()
    ring_sampler = subgraph_config["sampler"]
    min_ring_size = ring_sampler.get("min_ring_size", 3)
    max_ring_size = ring_sampler.get("max_ring_size", 7)

    n_heavy = max(2, round(target_n_nodes * HEAVY_ATOM_FRACTION))

    graph, elements, order_by_pair, n_degenerate = sample_skeleton_then_hydrogens(
        n_heavy,
        avg_degree,
        min_ring_size=min_ring_size,
        max_ring_size=max_ring_size,
    )
    distances, used_fallback = sample_bond_distances(graph, elements)
    src, dst = graph.edges()
    orders = np.array(
        [
            order_by_pair[(min(u, v), max(u, v))]
            for u, v in zip(src.tolist(), dst.tolist())
        ]
    )
    return graph, elements, distances, used_fallback, orders, n_degenerate


def plot_gallery(dist_config: dict, n_samples: int, seed: int, out_path: Path) -> None:
    n_cols = min(3, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for i in range(n_samples):
        graph, elements, distances, used_fallback, orders, n_degenerate = (
            sample_subgraph_with_chemistry(dist_config)
        )

        nx_graph = nx.Graph(dgl.to_networkx(graph))
        src, dst = graph.edges()
        edge_distance, edge_order = {}, {}
        for u, v, d, o in zip(src.tolist(), dst.tolist(), distances.tolist(), orders.tolist()):
            key = (min(u, v), max(u, v))
            edge_distance.setdefault(key, d)
            edge_order.setdefault(key, o)

        ax = axes[i]
        pos = nx.spring_layout(nx_graph, seed=seed + i, iterations=50)

        edge_list = list(nx_graph.edges())
        edge_d = [edge_distance[(min(u, v), max(u, v))] for u, v in edge_list]
        edge_o = [edge_order[(min(u, v), max(u, v))] for u, v in edge_list]
        norm = Normalize(vmin=min([0.9, *edge_d]), vmax=max([1.6, *edge_d]))
        edge_colors = plt.get_cmap("coolwarm")(norm(edge_d)) if edge_d else "gray"
        edge_widths = [1.2 * o for o in edge_o]

        nx.draw_networkx_edges(
            nx_graph, pos, ax=ax, edgelist=edge_list, edge_color=edge_colors, width=edge_widths
        )
        node_colors = [ELEMENT_COLORS[e] for e in elements[list(nx_graph.nodes())]]
        nx.draw_networkx_nodes(
            nx_graph,
            pos,
            ax=ax,
            node_size=90,
            node_color=node_colors,
            edgecolors="black",
            linewidths=0.5,
        )
        ax.set_axis_off()
        n_fallback = sum(used_fallback)
        degrees = graph.in_degrees()
        over_valence = sum(
            degrees[n].item() > VALENCE[elements[n]] for n in range(graph.num_nodes())
        )
        ax.set_title(
            f"n={graph.num_nodes()}, m={graph.num_edges() // 2}, "
            f"fallback_dist={n_fallback}/{len(used_fallback)}, "
            f"degenerate={n_degenerate}, over_valence={over_valence}",
            fontsize=8,
        )

    for j in range(n_samples, len(axes)):
        axes[j].set_axis_off()

    handles = [
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=c,
            markeredgecolor="black", label=e, markersize=8,
        )
        for e, c in ELEMENT_COLORS.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(ELEMENT_COLORS), fontsize=9)
    fig.suptitle(
        "valence-capped tree-with-rings + QM9-derived bond distances\n"
        "(node color = element, edge color = distance [blue=short, red=long], "
        "edge width = bond order)"
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(out_path, dpi=150)
    print(f"Saved gallery to {out_path}")


def plot_distance_validation(
    dist_config: dict, n_samples: int, seed: int, out_path: Path
) -> None:
    """Aggregate sampled distances per element pair across many sub-graphs,
    and check they match the target QM9 N(mu, sigma)."""
    per_pair_samples: dict[frozenset, list[float]] = {p: [] for p in BOND_PRIOR}
    fallback_samples: list[float] = []
    total_edges = 0

    for _ in range(n_samples):
        graph, elements, distances, used_fallback, _orders, _n_degenerate = (
            sample_subgraph_with_chemistry(dist_config)
        )
        src, dst = graph.edges()
        seen = set()
        for u, v, d, fb in zip(src.tolist(), dst.tolist(), distances.tolist(), used_fallback):
            key = (min(u, v), max(u, v))
            if key in seen:
                continue
            seen.add(key)
            total_edges += 1
            pair = frozenset({elements[u], elements[v]})
            (fallback_samples if fb else per_pair_samples[pair]).append(d)

    pairs = [p for p in BOND_PRIOR if per_pair_samples[p]]
    n_panels = len(pairs) + (1 if fallback_samples else 0)
    n_cols = 4
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    panel_idx = 0
    for pair in pairs:
        mu, sigma = BOND_PRIOR[pair]
        samples = np.array(per_pair_samples[pair])
        ax = axes[panel_idx]
        ax.hist(samples, bins=30, density=True, alpha=0.6, color="steelblue", label="sampled")
        xs = np.linspace(samples.min(), samples.max(), 200)
        ax.plot(
            xs,
            (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2),
            color="crimson",
            label="target N(mu,sigma)",
        )
        ax.axvline(mu, color="crimson", linestyle="--", linewidth=1)
        label = "-".join(sorted(pair))
        ax.set_title(f"{label} (n={len(samples)}, mean={samples.mean():.4f})", fontsize=9)
        ax.legend(fontsize=7)
        panel_idx += 1

    if fallback_samples:
        ax = axes[panel_idx]
        ax.hist(fallback_samples, bins=30, density=True, alpha=0.6, color="gray")
        ax.axvline(DEFAULT_BOND_PRIOR[0], color="black", linestyle="--", linewidth=1)
        ax.set_title(f"fallback pairs (n={len(fallback_samples)})", fontsize=9)
        panel_idx += 1

    for j in range(panel_idx, len(axes)):
        axes[j].set_axis_off()

    fig.suptitle(
        f"Sampled bond distances vs. QM9 target, aggregated over {n_samples} "
        f"sub-graphs ({total_edges} total edges)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    print(f"Saved distance-validation panel to {out_path}")


def plot_valence_validation(
    dist_config: dict, n_samples: int, seed: int, out_path: Path
) -> None:
    """Aggregate realized degree and realized valence usage per element
    across many sub-graphs, checked against each element's valence cap."""
    per_element_degree: dict[str, list[int]] = {e: [] for e in ELEMENTS}
    per_element_valence_used: dict[str, list[int]] = {e: [] for e in ELEMENTS}
    order_counts = {1: 0, 2: 0, 3: 0}
    total_degenerate = 0
    total_nodes = 0

    for _ in range(n_samples):
        graph, elements, _distances, _used_fallback, orders, n_degenerate = (
            sample_subgraph_with_chemistry(dist_config)
        )
        total_degenerate += n_degenerate
        total_nodes += graph.num_nodes()

        degrees = graph.in_degrees().tolist()
        for e, d in zip(elements, degrees):
            per_element_degree[e].append(d)

        src, dst = graph.edges()
        valence_used = np.zeros(graph.num_nodes(), dtype=np.int64)
        seen = set()
        for u, v, o in zip(src.tolist(), dst.tolist(), orders.tolist()):
            key = (min(u, v), max(u, v))
            if key in seen:
                continue
            seen.add(key)
            valence_used[u] += o
            valence_used[v] += o
            order_counts[int(o)] += 1
        for e, val in zip(elements, valence_used.tolist()):
            per_element_valence_used[e].append(val)

    fig, axes = plt.subplots(2, len(ELEMENTS), figsize=(4 * len(ELEMENTS), 7))

    for col, e in enumerate(ELEMENTS):
        cap = VALENCE[e]

        ax = axes[0, col]
        degrees = per_element_degree[e]
        if degrees:
            bins = np.arange(-0.5, max(degrees) + 1.5)
            ax.hist(degrees, bins=bins, color=ELEMENT_COLORS[e], edgecolor="black")
        ax.axvline(cap, color="red", linestyle="--", linewidth=1, label=f"valence cap={cap}")
        n_over = sum(d > cap for d in degrees)
        ax.set_title(f"{e} degree (n={len(degrees)}, >cap={n_over})", fontsize=9)
        ax.legend(fontsize=7)

        ax = axes[1, col]
        used = per_element_valence_used[e]
        if used:
            bins = np.arange(-0.5, max(used) + 1.5)
            ax.hist(used, bins=bins, color=ELEMENT_COLORS[e], edgecolor="black")
        ax.axvline(cap, color="red", linestyle="--", linewidth=1, label=f"valence cap={cap}")
        n_over = sum(u > cap for u in used)
        ax.set_title(f"{e} valence used (n={len(used)}, >cap={n_over})", fontsize=9)
        ax.legend(fontsize=7)

    total_bonds = sum(order_counts.values())
    order_str = ", ".join(
        f"order {o}: {c} ({c / total_bonds:.1%})" for o, c in order_counts.items()
    )
    fig.suptitle(
        f"Realized degree / valence usage by element, over {n_samples} sub-graphs "
        f"({total_nodes} nodes, {total_degenerate} degenerate attachments)\n{order_str}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=150)
    print(f"Saved valence-validation panel to {out_path}")
    print(f"  degenerate attachments: {total_degenerate} / {total_nodes} nodes")
    print(f"  bond order frequency: {order_str}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-gallery", type=int, default=6)
    parser.add_argument("--n-validation", type=int, default=300)
    parser.add_argument(
        "--gallery-out",
        type=str,
        default=str(Path(__file__).parent / "geometric_molecule_prior_gallery.png"),
    )
    parser.add_argument(
        "--distance-validation-out",
        type=str,
        default=str(Path(__file__).parent / "geometric_molecule_prior_validation.png"),
    )
    parser.add_argument(
        "--valence-validation-out",
        type=str,
        default=str(Path(__file__).parent / "geometric_molecule_prior_valence.png"),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dist_config = load_prior_distribution_config()

    plot_gallery(dist_config, args.n_gallery, args.seed, Path(args.gallery_out))
    plot_distance_validation(
        dist_config, args.n_validation, args.seed + 1000, Path(args.distance_validation_out)
    )
    plot_valence_validation(
        dist_config, args.n_validation, args.seed + 2000, Path(args.valence_validation_out)
    )


if __name__ == "__main__":
    main()
