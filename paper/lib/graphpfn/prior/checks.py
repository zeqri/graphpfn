# Modified for MolPFN (2026) from GraphPFN.
# Changes: Add configurable checks on the realized training fraction.
# See paper/LICENSE and paper/NOTICE at the repository root.

import torch

from lib.util import TaskType


class SanityCheckError(Exception):
    pass


def check_dataset(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_train_nodes: int,
    task_type: TaskType,
    min_features: int,
    n_classes: int | None,
    min_train_ratio: float = 0.0,
    max_train_ratio: float = 1.0,
) -> None:
    n_nodes = features.shape[0]

    if n_train_nodes >= n_nodes:
        raise SanityCheckError(
            f"n_train_nodes ({n_train_nodes}) must be < n_nodes ({n_nodes})"
        )

    check_train_ratio(n_train_nodes, n_nodes, min_train_ratio, max_train_ratio)

    check_no_nan(features, "features")
    check_no_nan(labels, "labels")
    check_min_features(features.shape[1], min_features)

    if task_type == TaskType.BINCLASS:
        check_n_classes(labels, 2)
        check_class_coverage(labels, n_train_nodes)
    elif task_type == TaskType.MULTICLASS:
        assert n_classes is not None
        check_n_classes(labels, n_classes)
        check_class_coverage(labels, n_train_nodes)


def check_train_ratio(
    n_train_nodes: int,
    n_nodes: int,
    min_train_ratio: float,
    max_train_ratio: float,
) -> None:
    """Reject datasets whose *realized* train ratio (n_train_nodes / actual
    n_nodes) drifted too far from what the sampled train_ratio distribution
    actually intended.

    n_train_nodes is computed upstream from a nominal/shared node count (see
    graph_then_attributes.py / attributes_then_graph.py), needed so every
    dataset in a DDP-sampled batch gets the exact same n_train_nodes (a
    single broadcast scalar, never scattered per-rank). When a dataset's
    actual node count differs a lot from that nominal count -- e.g. from
    size_jitter, per-sub-graph connectivity trimming, or (for
    molecule-skeleton) the emergent hydrogen count -- the realized ratio can
    drift far outside the sampled train_ratio's own range, producing
    barely-informative extremes: near-zero context (too few labeled examples
    to learn from, a noisy gradient) or near-total context (task nearly
    solved by copying neighbors, a redundant gradient). Defaults
    (0.0, 1.0) make this a no-op, since n_train_nodes < n_nodes is already
    guaranteed by the check above -- pass tighter bounds (e.g. matching the
    sampled train_ratio distribution's own [min, max]) to actually enforce
    this.
    """
    train_ratio_actual = n_train_nodes / n_nodes
    if train_ratio_actual < min_train_ratio or train_ratio_actual > max_train_ratio:
        raise SanityCheckError(
            f"Realized train_ratio ({train_ratio_actual:.3f}) outside "
            f"[{min_train_ratio}, {max_train_ratio}] "
            f"(n_train_nodes={n_train_nodes}, n_nodes={n_nodes})"
        )


def check_no_nan(tensor: torch.Tensor, name: str) -> None:
    if torch.isnan(tensor).any():
        raise SanityCheckError(f"{name} contains NaN values")
    if torch.isinf(tensor).any():
        raise SanityCheckError(f"{name} contains Inf values")


def check_min_features(n_features: int, min_features: int) -> None:
    if n_features < min_features:
        raise SanityCheckError(
            f"Only {n_features} features remain after dropping constants, "
            f"but min_features={min_features}"
        )


def check_n_classes(labels: torch.Tensor, n_classes: int) -> None:
    actual_n_classes = len(labels.unique())
    if actual_n_classes != n_classes:
        raise SanityCheckError(
            f"Expected {n_classes} classes, but got {actual_n_classes}"
        )


def check_class_coverage(labels: torch.Tensor, n_train_nodes: int) -> None:
    train_labels = labels[:n_train_nodes]
    test_labels = labels[n_train_nodes:]

    train_classes = set(train_labels.unique().tolist())
    test_classes = set(test_labels.unique().tolist())

    if not train_classes:
        raise SanityCheckError("No training examples")
    if not test_classes:
        raise SanityCheckError("No test examples")

    all_classes = train_classes | test_classes
    missing_in_train = all_classes - train_classes
    missing_in_test = all_classes - test_classes

    if missing_in_train:
        raise SanityCheckError(
            f"Classes {missing_in_train} present in test but not in train"
        )
    if missing_in_test:
        raise SanityCheckError(
            f"Classes {missing_in_test} present in train but not in test"
        )
