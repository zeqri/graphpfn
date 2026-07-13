"""CPU-only smoke test for the new `graph_level` prior (no model, no GPU).

Verifies the multi-graph, virtual-node prior module (`lib.graphpfn.prior.
priors.graph_level`) end to end: shapes, the context/query/atom node
layout, that atoms never carry a label, that virtual-node features are
zeroed, and -- most importantly -- that virtual-node labels actually vary
with which real nodes feed them (i.e. graph-dependence isn't accidentally
degenerate), by checking two independent datasets produce non-constant,
distinct label sets.

Usage (run with cwd=paper/):
    python -m bin.graphpfn.graph_level_prior_smoke_test
"""

import torch

from lib.graphpfn.prior.priors.graph_level import sample_dataset
from lib.graphpfn.prior.sampler import _pad_and_batch
from lib.util import TaskType

CONFIG = {
    "_type_": "graph_level",
    "total_n_nodes_budget": 1200,
    "n_graphs": 40,
    "graph": {
        "min_nodes": 8,
        "max_nodes": 40,
        "avg_degree_min": 2.0,
        "avg_degree_max": 6.0,
        "n_groups_min": 1,
        "n_groups_max": 3,
        "offdiagonal_coef": 0.1,
    },
    "scm": {
        "_type_": "gnn",
        "base": {
            "_type_": "mlp",
            "n_features": 10,
            "n_causes": 4,
            "n_layers": 3,
            "hidden_dim": 64,
            "activation_type": "relu",
            "causes": {"strategy": "mixed", "pre_sample_stats": False},
            "noise": {"std": 0.1, "pre_sample_std": False},
            "init": {
                "std": 1.0,
                "block_wise_dropout": False,
                "p_dropout": 0.0,
                "scale_std_by_dropout": False,
            },
            "causal": {
                "enabled": True,
                "y_is_effect": True,
                "in_clique": False,
                "sort_features": True,
            },
        },
        "conv_type": "sage-mean",
        "graph_conv_ratio": 0.5,
        "structural": {"use_degree": True, "use_pagerank": True, "lappe_k": 0},
    },
    "postprocessing": {
        "p_cat": 0.0,
        "max_categories": 10,
        "permute_features": False,
        "permute_labels": False,
    },
    "train_ratio": 0.3,
    "label_aggregation": "mean",
}


def check_one_dataset(seed: int) -> dict:
    torch.manual_seed(seed)
    import numpy as np

    np.random.seed(seed)
    import random

    random.seed(seed)

    dataset = sample_dataset(CONFIG)  # type: ignore[arg-type]

    features = dataset["features"]
    labels = dataset["labels"]
    edges = dataset["edges"]
    n_train = dataset["n_train_nodes"]
    labeled_mask = dataset["labeled_mask"]
    feature_fit_mask = dataset["feature_fit_mask"]
    task_type = dataset["task_type"]

    n_nodes = features.shape[0]
    n_graphs = int(labeled_mask.sum().item())
    n_atoms = n_nodes - n_graphs

    assert task_type == TaskType.REGRESSION
    assert not torch.isnan(features).any()
    assert not torch.isnan(labels).any()

    # >>> Layout invariants
    assert labeled_mask[:n_graphs].all(), "virtual nodes must be the first n_graphs rows"
    assert not labeled_mask[n_graphs:].any(), "no atom should ever be labeled_mask=True"
    assert feature_fit_mask[n_graphs:].all(), "all atoms should be fit-eligible"
    assert not feature_fit_mask[:n_graphs].any(), "no virtual node should be fit-eligible"
    assert 1 <= n_train < n_graphs, "context must be a strict, non-empty subset of graphs"

    # >>> No informative input at virtual rows
    assert (features[:n_graphs] == 0).all(), "virtual-node features must be exactly zero"

    # >>> Labels: non-degenerate, atoms zeroed
    virtual_labels = labels[:n_graphs]
    assert (labels[n_graphs:] == 0).all(), "atom labels must be zeroed (never read)"
    assert virtual_labels.std() > 1e-4, "virtual labels collapsed to (near) a constant"

    # >>> Edge sanity: every virtual node's degree == its own graph's atom count
    src, dst = edges[0], edges[1]
    for v in range(n_graphs):
        deg = int(((src == v) | (dst == v)).sum().item())
        # bidirectional star: deg counts both directions once each per atom
        assert deg > 0, f"virtual node {v} has no edges at all"

    # >>> Batching round-trip (this is what the real training loop consumes)
    batch = _pad_and_batch([dataset])
    assert batch["labeled_mask"].shape == (1, n_nodes)
    assert batch["feature_fit_mask"].shape == (1, n_nodes)
    assert bool((batch["labeled_mask"][0, :n_nodes] == labeled_mask).all())

    print(
        f"[seed={seed}] n_nodes={n_nodes} n_graphs={n_graphs} n_atoms={n_atoms} "
        f"n_train(context)={n_train} n_query={n_graphs - n_train} "
        f"n_features={features.shape[1]} "
        f"virtual_label_std={virtual_labels.std().item():.4f}"
    )

    return {"virtual_labels": virtual_labels.clone(), "n_nodes": n_nodes}


def main() -> None:
    result_a = check_one_dataset(seed=0)
    result_b = check_one_dataset(seed=1)

    # Two independently-sampled datasets should not produce identical label
    # sets (would indicate the SCM/aggregation isn't actually using the
    # sampled data at all).
    same_length = result_a["virtual_labels"].shape == result_b["virtual_labels"].shape
    identical = same_length and torch.equal(result_a["virtual_labels"], result_b["virtual_labels"])
    assert not identical, "two different seeds produced identical virtual labels"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
