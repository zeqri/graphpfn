"""Sample a *multi-graph* dataset from the new multi-graph prior and plot it.

Companion to dev/sample_prior_graph.py, but targets the new "multi-graph"
sampler (see exp/graphpfn/pretrain/multigraph/pretrain.toml and
lib/graphpfn/prior/graphs/multi_graph.py): instead of one large graph, a
single training "dataset" now combines `n_graphs` (2-4) independently-sampled
sub-graphs into one disjoint-union graph. Message passing / graph adapters
see a block-diagonal adjacency, so attention naturally stays scoped to each
node's own sub-graph -- nodes are colored by which sub-graph they belong to,
to make that structure visible.
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

CONFIG_PATH = PAPER_DIR / "exp/graphpfn/pretrain/multigraph/pretrain.toml"


def load_prior_distribution_config() -> dict:
    with open(CONFIG_PATH, "rb") as f:
        data = tomllib.load(f)
    return data["base_config"]["prior"]


def sample_multi_graph_with_membership(sampler_config: dict) -> tuple[dgl.DGLGraph, np.ndarray]:
    """Like lib.graphpfn.prior.graphs.multi_graph.sample_multi_graph, but also
    returns a per-node sub-graph id for visualization.

    Re-implemented locally (rather than reusing sample_multi_graph directly)
    so we can track which sub-graph each node came from -- this membership
    array is intentionally not part of the training-time PriorDataset, since
    the model must infer sub-graph boundaries only from the adjacency, not
    from an explicit id.
    """
    n_graphs = sampler_config["n_graphs"]
    sub_graph_template = sampler_config["sub_graph"]

    subgraphs = []
    membership = []
    for i in range(n_graphs):
        subgraph_config = resample_config(sub_graph_template)
        graph = sample_graph(subgraph_config)
        subgraphs.append(graph)
        membership.append(np.full(graph.num_nodes(), i))

    graph = dgl.batch(subgraphs)
    membership = np.concatenate(membership)
    return graph, membership


def graph_stats(nx_graph: nx.Graph) -> dict:
    n = nx_graph.number_of_nodes()
    m = nx_graph.number_of_edges()
    avg_degree = 2 * m / n if n else 0.0
    n_components = nx.number_connected_components(nx_graph)
    return {
        "n_nodes": n,
        "n_edges": m,
        "avg_degree": avg_degree,
        "n_components": n_components,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).parent / "multi_graph_prior_samples.png"),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dist_config = load_prior_distribution_config()

    n_cols = (args.n_samples + 1) // 2 if args.n_samples > 1 else 1
    n_rows = 2 if args.n_samples > 1 else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    palette = plt.get_cmap("tab10").colors

    for i in range(args.n_samples):
        sampled_config = sample_configs(dist_config, batch_size=1)[0]
        graph_config = sampled_config["prior"]["graph"]
        assert graph_config["sampler"]["_type_"] == "multi-graph", (
            "This script expects exp/graphpfn/pretrain/multigraph/pretrain.toml, "
            f"got sampler type {graph_config['sampler']['_type_']!r}"
        )

        graph, membership = sample_multi_graph_with_membership(graph_config["sampler"])
        # Skip the final global shuffle_nodes step that sample_graph would
        # normally apply here: it's only needed to decorrelate train/test
        # split from generation order during training, and would otherwise
        # just relabel node ids without changing what we want to show.
        nx_graph = nx.Graph(dgl.to_networkx(graph))

        stats = graph_stats(nx_graph)
        n_graphs = int(membership.max() + 1)
        print(f"[sample {i}] n_graphs={n_graphs} -> {stats}")

        ax = axes[i]
        pos = nx.spring_layout(nx_graph, seed=args.seed, iterations=50)
        nx.draw_networkx_edges(nx_graph, pos, ax=ax, alpha=0.15, width=0.4, edge_color="gray")

        node_colors = [palette[membership[n] % len(palette)] for n in nx_graph.nodes()]
        nx.draw_networkx_nodes(
            nx_graph, pos, ax=ax, node_size=8, node_color=node_colors, linewidths=0
        )
        ax.set_axis_off()
        ax.set_title(
            f"n_graphs={n_graphs}, n={stats['n_nodes']}, m={stats['n_edges']}, "
            f"avg_deg={stats['avg_degree']:.1f}, components={stats['n_components']}",
            fontsize=10,
        )

    for j in range(args.n_samples, len(axes)):
        axes[j].set_axis_off()

    fig.suptitle(
        "Samples from the multi-graph prior "
        "(2-4 independent sub-graphs combined per dataset, colored by sub-graph)"
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
