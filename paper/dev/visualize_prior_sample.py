"""Sample a full dataset (graph + features + labels + train/test split) from
the currently-training prior and visualize it end-to-end.

Unlike dev/sample_prior_graph.py / dev/sample_multi_graph_prior.py (which only
look at the graph *structure*), this runs the exact same pipeline used during
training (lib.graphpfn.prior.priors.sample_dataset via the retry wrapper in
lib.graphpfn.prior.sampler) to get a real PriorDataset: node features, labels,
edges, and the train/test split the model actually sees. For each sample this
plots:
  - the graph, colored by train (green) vs. query/test (gray) node membership
    -- matching Figure 1's legend in the paper -- and, for the multi-graph
    prior, annotated with how many sub-graphs went into it
  - the label distribution (histogram for regression, bar chart for
    classification)

Usage:
    python dev/visualize_prior_sample.py --config multigraph --n-samples 4
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

from lib.graphpfn.prior.config import sample_configs  # noqa: E402
from lib.graphpfn.prior.sampler import _sample_dataset_with_retry  # noqa: E402
from lib.util import TaskType  # noqa: E402


def load_prior_distribution_config(config_name: str) -> dict:
    path = PAPER_DIR / f"exp/graphpfn/pretrain/{config_name}/pretrain.toml"
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return data["base_config"]["prior"]


def plot_sample(fig, gs_graph, gs_labels, dist_config: dict, seed: int) -> None:
    sampled_config = sample_configs(dist_config, batch_size=1)[0]
    dataset = _sample_dataset_with_retry(sampled_config)

    features = dataset["features"]
    labels = dataset["labels"].numpy()
    edges = dataset["edges"]
    n_train_nodes = dataset["n_train_nodes"]
    task_type = dataset["task_type"]
    n_nodes = features.shape[0]

    graph_config = sampled_config["prior"]["graph"]
    sampler_type = graph_config["sampler"]["_type_"]
    n_graphs = graph_config["sampler"].get("n_graphs", 1)

    graph = dgl.graph((edges[0], edges[1]), num_nodes=n_nodes)
    nx_graph = nx.Graph(dgl.to_networkx(graph))

    is_train = np.zeros(n_nodes, dtype=bool)
    is_train[:n_train_nodes] = True  # split is contiguous: train=[0, n_train_nodes)

    ax_graph = fig.add_subplot(gs_graph)
    pos = nx.spring_layout(nx_graph, seed=seed, iterations=50)
    nx.draw_networkx_edges(nx_graph, pos, ax=ax_graph, alpha=0.15, width=0.4, edge_color="gray")
    node_colors = ["mediumseagreen" if is_train[n] else "lightgray" for n in nx_graph.nodes()]
    nx.draw_networkx_nodes(
        nx_graph, pos, ax=ax_graph, node_size=8, node_color=node_colors, linewidths=0
    )
    ax_graph.set_axis_off()
    ax_graph.set_title(
        f"sampler={sampler_type} (n_graphs={n_graphs}), n={n_nodes}, "
        f"n_train={n_train_nodes}, n_features={features.shape[1]}, task={task_type.name}",
        fontsize=9,
    )

    ax_labels = fig.add_subplot(gs_labels)
    if task_type == TaskType.REGRESSION:
        ax_labels.hist(labels, bins=30, color="steelblue")
        ax_labels.set_xlabel("label value")
        ax_labels.set_ylabel("count")
    else:
        values, counts = np.unique(labels, return_counts=True)
        ax_labels.bar(values.astype(int).astype(str), counts, color="steelblue")
        ax_labels.set_xlabel("class")
        ax_labels.set_ylabel("count")
    ax_labels.set_title("label distribution (all nodes)", fontsize=9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="multigraph",
        help="Name of the experiment dir under exp/graphpfn/pretrain/ (e.g. 'multigraph' or 'main').",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path (default: dev/prior_sample_<config>.png).",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_path = args.out or str(Path(__file__).parent / f"prior_sample_{args.config}.png")

    dist_config = load_prior_distribution_config(args.config)

    fig = plt.figure(figsize=(11, 4.5 * args.n_samples))
    gs = fig.add_gridspec(args.n_samples, 2, width_ratios=[2, 1])

    for i in range(args.n_samples):
        print(f"Sampling dataset {i}...")
        plot_sample(fig, gs[i, 0], gs[i, 1], dist_config, seed=args.seed + i)

    fig.suptitle(f"Full dataset samples from prior config: {args.config}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
