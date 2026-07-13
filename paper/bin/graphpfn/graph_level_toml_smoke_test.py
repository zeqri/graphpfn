"""Validates exp/graphpfn/pretrain/graph_level/pretrain.toml end to end:
parses it, runs it through the real `sample_configs` distribution resolver
(exactly what `GraphPriorSampler`/`sample_batch` do at actual pretraining
time), and feeds the resulting concrete config into `sample_dataset` +
`_pad_and_batch`, with the same invariants checked by
graph_level_prior_smoke_test.py.

Usage (run with cwd=paper/):
    python -m bin.graphpfn.graph_level_toml_smoke_test
"""

import tomllib
from pathlib import Path

from lib.graphpfn.prior.config import sample_configs
from lib.graphpfn.prior.priors import sample_dataset
from lib.graphpfn.prior.sampler import _pad_and_batch
from lib.util import TaskType

TOML_PATH = Path("exp/graphpfn/pretrain/graph_level/pretrain.toml")


def main() -> None:
    with open(TOML_PATH, "rb") as f:
        toml_config = tomllib.load(f)

    prior_distribution_config = toml_config["base_config"]["prior"]

    for i in range(3):
        sampled = sample_configs(prior_distribution_config, batch_size=1)[0]
        dataset = sample_dataset(sampled["prior"])  # type: ignore[arg-type]

        features = dataset["features"]
        labels = dataset["labels"]
        labeled_mask = dataset["labeled_mask"]
        feature_fit_mask = dataset["feature_fit_mask"]
        n_train = dataset["n_train_nodes"]

        n_nodes = features.shape[0]
        n_graphs = int(labeled_mask.sum().item())

        assert dataset["task_type"] == TaskType.REGRESSION
        assert labeled_mask[:n_graphs].all()
        assert not labeled_mask[n_graphs:].any()
        assert not feature_fit_mask[:n_graphs].any()
        assert (features[:n_graphs] == 0).all()
        assert 1 <= n_train < n_graphs
        assert labels[:n_graphs].std().item() > 1e-4

        batch = _pad_and_batch([dataset])
        assert batch["labeled_mask"].shape == (1, n_nodes)

        print(
            f"[draw {i}] n_nodes={n_nodes} n_graphs={n_graphs} "
            f"n_features={features.shape[1]} n_train(context)={n_train} "
            f"scm.graph_conv_ratio={sampled['prior']['scm']['graph_conv_ratio']} "
            f"scm.conv_type={sampled['prior']['scm']['conv_type']}"
        )

    print("\nTOML round-trip: all checks passed.")


if __name__ == "__main__":
    main()
