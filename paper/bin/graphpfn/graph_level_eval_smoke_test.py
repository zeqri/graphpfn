"""CPU-only smoke test for the periodic-eval path adaptation.

Exercises exactly what `evaluate()` -> `get_synthetic_eval_dataset()` does at
real pretraining time (`unbatch_prior_dataset` + `convert_to_graph_dataset`),
for both the new `graph_level` prior and an existing node-level prior, and
checks:
  - graph_level: the "test" mask contains only query virtual nodes (no
    atoms), and `drop_constant_features` fit on `feature_fit_mask` (atoms)
    does NOT collapse to zero columns -- while fitting on the old, buggy
    `masks["train"]` (context virtual rows, all-zero placeholders) WOULD.
  - node-level (graph_then_attributes): behavior is unchanged -- "test"
    mask matches the old `idx >= n_train` definition exactly, since
    labeled_mask is all-True there.
  - the label-double-standardization fix: graph_level datasets carry
    labels_standardized=True, so `prepare_labels` (called by
    evaluate_dataset) must skip re-standardization entirely (a no-op);
    node-level datasets carry labels_standardized=False, so `prepare_labels`
    must still standardize exactly as before.

Usage (run with cwd=paper/):
    python -m bin.graphpfn.graph_level_eval_smoke_test
"""

import tomllib
from pathlib import Path

import numpy as np
import torch

import lib.graph.data
from lib.graphpfn.prior.config import sample_configs
from lib.graphpfn.prior.priors import sample_dataset
from lib.graphpfn.prior.priors.graph_level import sample_dataset as sample_graph_level
from lib.graphpfn.prior.sampler import _pad_and_batch
from lib.graphpfn.prior.util import convert_to_graph_dataset, unbatch_prior_dataset

GRAPH_LEVEL_CONFIG = {
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

NODE_LEVEL_TOML_PATH = Path("exp/graphpfn/pretrain/main_loss_comparison/pretrain.toml")


def check_graph_level() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    import random

    random.seed(0)

    dataset = sample_graph_level(GRAPH_LEVEL_CONFIG)  # type: ignore[arg-type]
    batch = _pad_and_batch([dataset])
    [unbatched] = unbatch_prior_dataset(batch)
    graph_dataset = convert_to_graph_dataset(unbatched, "synthetic-graph-level-000")

    masks = graph_dataset.data["masks"]
    n_nodes = graph_dataset.data["labels"].shape[0]
    n_graphs = int(unbatched["labeled_mask"].sum().item())
    n_train = unbatched["n_train_nodes"]

    # >>> "test" must be exactly the query virtual nodes: [n_train, n_graphs)
    expected_test = np.zeros(n_nodes, dtype=bool)
    expected_test[n_train:n_graphs] = True
    assert np.array_equal(masks["test"], expected_test), (
        "test mask must contain only query virtual nodes, no atoms"
    )
    assert masks["test"].sum() == n_graphs - n_train
    # No atom (index >= n_graphs) is ever in "test".
    assert not masks["test"][n_graphs:].any()

    # >>> feature_fit_mask must be present and point at atoms only.
    fit_mask = graph_dataset.data["feature_fit_mask"]
    assert fit_mask is not None
    assert not fit_mask[:n_graphs].any()
    assert fit_mask[n_graphs:].all()

    # >>> The actual fix: drop_constant_features on feature_fit_mask keeps
    # real columns; the old behavior (fit on masks["train"], i.e. all-zero
    # context virtual rows) would collapse every column to "constant" and
    # drop them all.
    features = torch.as_tensor(graph_dataset.data["num_features"])
    fit_mask_t = torch.as_tensor(fit_mask)
    train_mask_t = torch.as_tensor(masks["train"])

    n_cols_before = features.shape[1]
    fixed = lib.graph.data.drop_constant_features(features, fit_mask_t)
    buggy = lib.graph.data.drop_constant_features(features, train_mask_t)

    assert fixed.shape[1] == n_cols_before, (
        f"fitting on atoms should keep all {n_cols_before} columns, "
        f"got {fixed.shape[1]}"
    )
    assert buggy.shape[1] == 0, (
        "sanity check: fitting on all-zero context-virtual rows should "
        f"collapse every column (old bug) -- got {buggy.shape[1]} survivors, "
        "expected 0. If this fails, the 'bug' being fixed no longer exists "
        "and this regression test should be revisited."
    )

    print(
        f"[graph_level] n_nodes={n_nodes} n_graphs={n_graphs} n_train={n_train} "
        f"n_test(query)={masks['test'].sum()} "
        f"drop_constant_features: fit-on-atoms keeps {fixed.shape[1]}/{n_cols_before} cols, "
        f"fit-on-train(buggy) keeps {buggy.shape[1]}/{n_cols_before} cols"
    )

    # >>> Double-standardization fix: labels_standardized must be True, and
    # evaluate_dataset's `prepare_labels(dataset, not already_standardized)`
    # call must therefore be a no-op (returns None, leaves labels untouched)
    # -- this is exactly the mechanism that fixed the mean R2 gap measured
    # against the real trained checkpoint (0.0083 -> 0.1227, see
    # graph_level_label_standardization_validation.py).
    assert graph_dataset.data.get("labels_standardized") is True
    already_standardized = graph_dataset.data.get("labels_standardized", False)
    labels_before = graph_dataset.data["labels"].copy()
    stats = lib.graph.data.prepare_labels(graph_dataset, not already_standardized)
    assert stats is None, "graph_level eval must not re-standardize labels"
    assert np.array_equal(graph_dataset.data["labels"], labels_before), (
        "prepare_labels must not mutate already-standardized graph_level labels"
    )
    print("[graph_level] prepare_labels correctly skipped re-standardization")


def check_node_level_unchanged() -> None:
    with open(NODE_LEVEL_TOML_PATH, "rb") as f:
        toml_config = tomllib.load(f)
    prior_distribution_config = toml_config["base_config"]["prior"]

    torch.manual_seed(1)
    np.random.seed(1)

    sampled = sample_configs(prior_distribution_config, batch_size=1)[0]
    # Force regression: task._type_ is itself a random choice
    # (regression/binclass/multiclass) in this toml, but this test
    # specifically checks the regression-only prepare_labels path.
    sampled["prior"]["task"]["_type_"] = "regression"  # type: ignore[index]
    dataset = sample_dataset(sampled["prior"])  # type: ignore[arg-type]
    assert bool(dataset["labeled_mask"].all()), (
        "node-level priors must keep labeled_mask all-True"
    )

    batch = _pad_and_batch([dataset])
    [unbatched] = unbatch_prior_dataset(batch)
    graph_dataset = convert_to_graph_dataset(unbatched, "synthetic-node-level-000")

    masks = graph_dataset.data["masks"]
    n_nodes = graph_dataset.data["labels"].shape[0]
    n_train = unbatched["n_train_nodes"]

    expected_test = np.arange(n_nodes) >= n_train
    assert np.array_equal(masks["test"], expected_test), (
        "node-level test mask must be unchanged: idx >= n_train"
    )

    fit_mask = graph_dataset.data["feature_fit_mask"]
    assert fit_mask is not None  # always populated by convert_to_graph_dataset now

    print(
        f"[node-level] n_nodes={n_nodes} n_train={n_train} "
        f"n_test={masks['test'].sum()} (== idx>=n_train, as before) "
        f"feature_fit_mask present, sum={int(fit_mask.sum())}"
    )

    # >>> labels_standardized must be False for node-level priors, so
    # evaluate_dataset's re-standardization behavior is exactly unchanged.
    assert graph_dataset.data.get("labels_standardized", False) is False
    already_standardized = graph_dataset.data.get("labels_standardized", False)
    labels_before = graph_dataset.data["labels"].copy()
    stats = lib.graph.data.prepare_labels(graph_dataset, not already_standardized)
    assert stats is not None, "node-level eval must still standardize labels"
    assert not np.array_equal(graph_dataset.data["labels"], labels_before), (
        "prepare_labels must still mutate node-level labels (unchanged behavior)"
    )
    print("[node-level] prepare_labels still standardizes as before")


def main() -> None:
    check_graph_level()
    check_node_level_unchanged()
    print("\nAll eval-path checks passed.")


if __name__ == "__main__":
    main()
