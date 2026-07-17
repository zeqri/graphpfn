"""Sample from the "organic molecule"-like multigraph prior and visualize it.

Targets exp/graphpfn/pretrain/multigraph_molecule/pretrain.toml specifically
(unlike dev/sample_multi_graph_prior.py, which targets the original
multigraph config with few/large sub-graphs). This prior combines many
(10-30) small, sparse, similarly-sized sub-graphs per dataset -- so alongside
the graph layout (colored by sub-graph membership), this also plots each
sub-graph's actual node count against the dataset's shared target size
(base_n_nodes +/- size_jitter), to make the "many small graphs, but
consistent size within one dataset" property directly visible.

--sampler lets you swap the sub-graph structural generator for a controlled
comparison, without creating a separate training config:
  - "multi-level-sbm-with-pa" (default, matches the actual training config)
  - "sbm": no hierarchical first/second-level combination, no preferential
    attachment -- just a single degree-corrected SBM per sub-graph, same
    n_groups=2 / offdiagonal_coef=0.1.
  - "geometric": nodes are points in latent space (n_latent_features=2),
    edges connect nearby points -- no community/block structure at all, a
    completely different generative family from the (degree-corrected) SBM
    ones above.
  - "tree-with-rings": a degree-capped random tree (connected by
    construction, max_degree=4) plus a small number of ring-closing edges
    between nodes already close in tree-distance (ring size 3-7) -- a
    molecule-like alternative that sidesteps the size-consistency problems
    the other samplers hit at this scale, since it never needs
    extract_largest_component to fix up disconnected pieces.
In all cases, n_graphs / base_n_nodes / size_jitter / avg_degree stay
identical, so any visual/structural difference comes purely from the
structural generator itself.
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
from lib.graphpfn.prior.graphs import sample_graph  # noqa: E402

CONFIG_PATH = PAPER_DIR / "exp/graphpfn/pretrain/multigraph_molecule/pretrain.toml"


def load_prior_distribution_config(sampler: str) -> dict:
    with open(CONFIG_PATH, "rb") as f:
        data = tomllib.load(f)
    dist_config = data["base_config"]["prior"]

    if sampler == "sbm":
        # Swap the sub-graph structural generator from multi-level-sbm-with-pa
        # to a plain single-level degree-corrected SBM, keeping n_groups /
        # offdiagonal_coef the same for a controlled comparison. Everything
        # else (n_graphs, base_n_nodes, size_jitter, avg_degree) is untouched.
        multi_graph_config = dist_config["prior"]["values"][0]["graph"]["sampler"]["values"][0]
        multi_graph_config["sub_graph"]["sampler"] = {
            "_distribution_": "choice",
            "_runtime_": True,
            "values": [
                {
                    "_type_": "sbm",
                    "n_groups": 2,
                    "offdiagonal_coef": 0.1,
                }
            ],
        }
    elif sampler == "geometric":
        # Swap to a random geometric graph: nodes are points in latent space,
        # edges connect nearby points. No block/community structure at all --
        # a different generative family, not just a simplification of SBM.
        multi_graph_config = dist_config["prior"]["values"][0]["graph"]["sampler"]["values"][0]
        multi_graph_config["sub_graph"]["sampler"] = {
            "_distribution_": "choice",
            "_runtime_": True,
            "values": [
                {
                    "_type_": "geometric",
                    "n_latent_features": 2,
                }
            ],
        }
    elif sampler == "tree-with-rings":
        # Swap to the molecule-like degree-capped tree + ring-closure
        # sampler (lib.graphpfn.prior.graphs.tree_with_rings).
        multi_graph_config = dist_config["prior"]["values"][0]["graph"]["sampler"]["values"][0]
        multi_graph_config["sub_graph"]["sampler"] = {
            "_distribution_": "choice",
            "_runtime_": True,
            "values": [
                {
                    "_type_": "tree-with-rings",
                    "max_degree": 4,
                    "min_ring_size": 3,
                    "max_ring_size": 7,
                }
            ],
        }
    elif sampler != "multi-level-sbm-with-pa":
        raise ValueError(f"Unknown --sampler: {sampler!r}")

    return dist_config


def sample_with_membership(sampler_config: dict):
    """Re-implements sample_multi_graph, but also returns per-sub-graph node
    counts and a per-node membership array for visualization -- deliberately
    not part of the training-time PriorDataset.
    """
    n_graphs = sampler_config["n_graphs"]
    sub_graph_template = sampler_config["sub_graph"]
    base_n_nodes = sampler_config.get("base_n_nodes")
    size_jitter = sampler_config.get("size_jitter", 0.0)

    subgraphs, membership, sizes = [], [], []
    for i in range(n_graphs):
        subgraph_config = resample_config(sub_graph_template)
        if base_n_nodes is not None:
            jitter = (
                1.0 + np.random.uniform(-size_jitter, size_jitter) if size_jitter else 1.0
            )
            subgraph_config["n_nodes"] = max(int(round(base_n_nodes * jitter)), 4)
        graph = sample_graph(subgraph_config)
        subgraphs.append(graph)
        membership.append(np.full(graph.num_nodes(), i))
        sizes.append(graph.num_nodes())

    merged = dgl.batch(subgraphs)
    membership = np.concatenate(membership)
    return merged, membership, sizes, base_n_nodes, size_jitter


def plot_sample(fig, gs_graph, gs_sizes, dist_config: dict, seed: int) -> None:
    sampled_config = sample_configs(dist_config, batch_size=1)[0]
    graph_config = sampled_config["prior"]["graph"]
    assert graph_config["sampler"]["_type_"] == "multi-graph"

    graph, membership, sizes, base_n_nodes, size_jitter = sample_with_membership(
        graph_config["sampler"]
    )
    nx_graph = nx.Graph(dgl.to_networkx(graph))
    n_graphs = len(sizes)

    print(
        f"n_graphs={n_graphs} base_n_nodes={base_n_nodes} "
        f"total_nodes={graph.num_nodes()} sizes={sorted(sizes)}"
    )

    ax_graph = fig.add_subplot(gs_graph)
    pos = nx.spring_layout(nx_graph, seed=seed, iterations=50)
    nx.draw_networkx_edges(nx_graph, pos, ax=ax_graph, alpha=0.15, width=0.4, edge_color="gray")
    cmap = plt.get_cmap("turbo")
    colors = [cmap(i / max(n_graphs - 1, 1)) for i in range(n_graphs)]
    node_colors = [colors[membership[n]] for n in nx_graph.nodes()]
    nx.draw_networkx_nodes(
        nx_graph, pos, ax=ax_graph, node_size=10, node_color=node_colors, linewidths=0
    )
    ax_graph.set_axis_off()
    ax_graph.set_title(
        f"n_graphs={n_graphs}, total_nodes={graph.num_nodes()}, base_n_nodes={base_n_nodes}",
        fontsize=9,
    )

    ax_sizes = fig.add_subplot(gs_sizes)
    sizes_sorted = sorted(sizes)
    bar_colors = [colors[i] for i in np.argsort(sizes)]
    ax_sizes.bar(range(n_graphs), sizes_sorted, color=bar_colors)
    if base_n_nodes is not None:
        ax_sizes.axhline(base_n_nodes, color="black", linestyle="--", linewidth=1, label="base_n_nodes")
        if size_jitter:
            ax_sizes.axhspan(
                base_n_nodes * (1 - size_jitter),
                base_n_nodes * (1 + size_jitter),
                color="black",
                alpha=0.08,
                label=f"+/-{size_jitter:.0%} jitter",
            )
        ax_sizes.legend(fontsize=7, loc="upper left")
    ax_sizes.set_xlabel("sub-graph (sorted by size)")
    ax_sizes.set_ylabel("n_nodes")
    ax_sizes.set_title("sub-graph size consistency", fontsize=9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument(
        "--sampler",
        type=str,
        default="multi-level-sbm-with-pa",
        choices=["multi-level-sbm-with-pa", "sbm", "geometric", "tree-with-rings"],
        help="Sub-graph structural generator. 'sbm' drops the hierarchical "
        "first/second-level combination and preferential attachment, using a "
        "plain single-level degree-corrected SBM instead (same n_groups/"
        "offdiagonal_coef), for a controlled comparison.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path (default: dev/molecule_prior_samples_<sampler>.png).",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_path = args.out or str(
        Path(__file__).parent / f"molecule_prior_samples_{args.sampler}.png"
    )

    dist_config = load_prior_distribution_config(args.sampler)

    fig = plt.figure(figsize=(11, 4.5 * args.n_samples))
    gs = fig.add_gridspec(args.n_samples, 2, width_ratios=[2, 1])

    for i in range(args.n_samples):
        print(f"[sample {i}]", end=" ")
        plot_sample(fig, gs[i, 0], gs[i, 1], dist_config, seed=args.seed + i)

    fig.suptitle(
        f"Samples from the multigraph_molecule prior (sub-graph sampler: {args.sampler})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
