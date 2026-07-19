"""Sample from the valence-aware molecule-skeleton prior and visualize it.

Targets exp/graphpfn/pretrain/multigraph_molecule_valence/pretrain.toml --
the training config wired up to lib.graphpfn.prior.graphs.molecule_skeleton
(per-element C/N/O/F valence caps + sampled bond order, topology only, no
distances yet), as opposed to dev/visualize_molecule_prior.py's
tree-with-rings (one uniform max_degree for every node).

Unlike molecule_skeleton.sample_molecule_skeleton (the production sampler,
which only returns a plain graph -- element identity is used internally to
shape connectivity but isn't retained anywhere), this script reimplements
just the thin hydrogen-attachment wrapper locally so it can also return a
per-node element label for visualization, while still calling the exact same
production tree/ring-growth routine
(_sample_valence_capped_tree_with_rings) -- so what you see here is exactly
what training samples, just with extra instrumentation.

For each sampled multi-graph dataset, plots:
  - the graph layout, colored by element (H/C/N/O/F), sized/shaped the same
    across all sub-graphs in the figure so composition is easy to compare
  - realized degree per element vs. that element's valence cap, aggregated
    over the whole dataset (all sub-graphs combined) -- validates that the
    valence-capping is doing its job through the *actual* multi-graph +
    molecule-skeleton pipeline used in training, not just the standalone
    prototype in sample_geometric_molecule_prior.py

Usage:
    python dev/visualize_molecule_skeleton_prior.py --n-samples 4
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

from lib.graphpfn.prior.config import resample_config, sample_configs  # noqa: E402
from lib.graphpfn.prior.graphs.molecule_skeleton import (  # noqa: E402
    HEAVY_ELEMENT_WEIGHTS,
    HEAVY_ELEMENTS,
    VALENCE,
    _sample_valence_capped_tree_with_rings,
)
from lib.graphpfn.prior.graphs.util import to_simple  # noqa: E402

CONFIG_PATH = PAPER_DIR / "exp/graphpfn/pretrain/multigraph_molecule_valence/pretrain.toml"

ELEMENTS = ["H", "C", "N", "O", "F"]
ELEMENT_COLORS = {
    "H": "#dddddd",
    "C": "#333333",
    "N": "#3060f0",
    "O": "#e03030",
    "F": "#30c030",
}


def load_prior_distribution_config() -> dict:
    with open(CONFIG_PATH, "rb") as f:
        data = tomllib.load(f)
    return data["base_config"]["prior"]


def sample_skeleton_with_elements(
    n_nodes: int,
    avg_degree: float,
    *,
    heavy_atom_fraction: float,
    min_ring_size: int,
    max_ring_size: int,
) -> tuple[dgl.DGLGraph, np.ndarray]:
    """Same algorithm as molecule_skeleton.sample_molecule_skeleton (calls the
    exact same production tree/ring-growth routine), but also returns a
    per-node element label for visualization."""
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
    remaining = np.clip(heavy_valence_cap - valence_used, 0, None)

    edges = list(order_by_pair.keys())
    next_id = n_heavy
    for i in range(n_heavy):
        for _ in range(int(remaining[i])):
            edges.append((i, next_id))
            next_id += 1
    n_total = next_id

    elements = np.concatenate([heavy_elements, np.full(n_total - n_heavy, "H")])

    src = torch.tensor([e[0] for e in edges] + [e[1] for e in edges], dtype=torch.int64)
    dst = torch.tensor([e[1] for e in edges] + [e[0] for e in edges], dtype=torch.int64)
    graph = to_simple(dgl.graph((src, dst), num_nodes=n_total))
    return graph, elements


def sample_multi_graph_with_elements(sampler_config: dict):
    """Mirrors lib.graphpfn.prior.graphs.multi_graph.sample_multi_graph, but
    also returns a per-node element label for visualization."""
    n_graphs = sampler_config["n_graphs"]
    sub_graph_template = sampler_config["sub_graph"]
    base_n_nodes = sampler_config["base_n_nodes"]
    size_jitter = sampler_config.get("size_jitter", 0.0)

    subgraphs, all_elements = [], []
    for _ in range(n_graphs):
        subgraph_config = resample_config(sub_graph_template)
        jitter = 1.0 + np.random.uniform(-size_jitter, size_jitter) if size_jitter else 1.0
        n_nodes = max(int(round(base_n_nodes * jitter)), 4)
        skeleton_kwargs = {
            k: v for k, v in subgraph_config["sampler"].items() if not k.startswith("_")
        }
        graph, elements = sample_skeleton_with_elements(
            n_nodes, subgraph_config["avg_degree"], **skeleton_kwargs
        )
        subgraphs.append(graph)
        all_elements.append(elements)

    merged = dgl.batch(subgraphs)
    return merged, np.concatenate(all_elements)


def plot_sample(fig, gs_graph, gs_degree, dist_config: dict, seed: int) -> None:
    sampled_config = sample_configs(dist_config, batch_size=1)[0]
    graph_config = sampled_config["prior"]["graph"]
    assert graph_config["sampler"]["_type_"] == "multi-graph"

    graph, elements = sample_multi_graph_with_elements(graph_config["sampler"])
    nx_graph = nx.Graph(dgl.to_networkx(graph))
    n_graphs = graph_config["sampler"]["n_graphs"]
    base_n_nodes = graph_config["sampler"]["base_n_nodes"]

    degrees = np.array(list(dict(nx_graph.degree()).values()))
    composition = {e: int((elements == e).sum()) for e in ELEMENTS}
    print(
        f"n_graphs={n_graphs} base_n_nodes={base_n_nodes} total_nodes={graph.num_nodes()} "
        f"composition={composition}"
    )

    ax_graph = fig.add_subplot(gs_graph)
    pos = nx.spring_layout(nx_graph, seed=seed, iterations=50)
    nx.draw_networkx_edges(nx_graph, pos, ax=ax_graph, alpha=0.15, width=0.4, edge_color="gray")
    node_colors = [ELEMENT_COLORS[elements[n]] for n in nx_graph.nodes()]
    nx.draw_networkx_nodes(
        nx_graph, pos, ax=ax_graph, node_size=8, node_color=node_colors, linewidths=0
    )
    ax_graph.set_axis_off()
    comp_str = ", ".join(f"{e}={c}" for e, c in composition.items())
    ax_graph.set_title(
        f"n_graphs={n_graphs}, n={graph.num_nodes()}, base_n_nodes={base_n_nodes}\n{comp_str}",
        fontsize=9,
    )

    ax_degree = fig.add_subplot(gs_degree)
    n_elements = len(ELEMENTS)
    width = 0.8 / n_elements
    max_degree_seen = max(int(degrees.max()), max(VALENCE.values())) if len(degrees) else 0
    for i, e in enumerate(ELEMENTS):
        e_degrees = degrees[elements == e]
        cap = VALENCE[e]
        n_over_cap = int((e_degrees > cap).sum())
        counts = np.bincount(e_degrees, minlength=max_degree_seen + 1)
        xs = np.arange(len(counts)) + i * width
        ax_degree.bar(
            xs, counts, width=width, color=ELEMENT_COLORS[e],
            label=f"{e} (cap={cap}, n={len(e_degrees)}, >cap={n_over_cap})",
            edgecolor="black", linewidth=0.3,
        )
    ax_degree.set_xticks(np.arange(max_degree_seen + 1) + 0.4)
    ax_degree.set_xticklabels(np.arange(max_degree_seen + 1))
    ax_degree.set_xlabel("degree")
    ax_degree.set_ylabel("count")
    ax_degree.set_title("degree by element", fontsize=9)
    ax_degree.legend(fontsize=6.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).parent / "molecule_skeleton_prior_samples.png"),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dist_config = load_prior_distribution_config()

    fig = plt.figure(figsize=(11, 4.5 * args.n_samples))
    gs = fig.add_gridspec(args.n_samples, 2, width_ratios=[2, 1])

    for i in range(args.n_samples):
        plot_sample(fig, gs[i, 0], gs[i, 1], dist_config, seed=args.seed + i)

    fig.suptitle(
        "Samples from the molecule-skeleton prior "
        "(per-element valence caps + sampled bond order, topology only)"
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
