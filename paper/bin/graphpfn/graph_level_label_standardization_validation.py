"""Validates whether the label-double-standardization bug actually moves the
"synthetic" R2 score, using the REAL trained checkpoint from the currently
running job -- not a random-init model.

Background: `evaluate_dataset` (bin/graphpfn/pretrain.py) calls
`lib.graph.data.prepare_labels(dataset, True)`, which re-standardizes labels
using ONLY the train (context) subset's own mean/std. For the graph_level
prior, labels are ALREADY standardized once at generation time
(graph_level.py's own `standard_scaling`, fit over ALL virtual nodes --
context + query together). So graph_level eval labels get standardized
TWICE, the second time on a much smaller, sometimes-tiny (as small as 1
node) subset, with no epsilon floor on the std (unlike the prior's own
`standard_scaling`, which clamps to 1e-6) -- risking a division by (near)
zero.

This script does NOT touch the library code. It runs two evaluation paths
side by side, on the SAME sampled synthetic datasets, using the SAME real
checkpoint weights:
  - "buggy":  the actual, current `evaluate_dataset` (imported unmodified
    from bin.graphpfn.pretrain) -- exactly what the live training run uses.
  - "fixed":  a local copy of `evaluate_dataset` with the re-standardization
    step removed entirely (labels are used as-is; predictions are compared
    directly with no inverse-transform), since graph_level labels are
    already on a consistent, comparable scale.

It reports, per sampled dataset: n_train (context size), whether it hit the
n_train==1 degenerate case, and the buggy vs. fixed R2 score -- plus
aggregate means matching how the real "synthetic" metric is computed
(mean over draws).

Usage (run with cwd=paper/, needs the env active: see run_full.sh for the
module-load / venv-activate incantation):
    python -m bin.graphpfn.graph_level_label_standardization_validation \
        --n-datasets 50

Does not require a GPU (falls back to CPU automatically), but will use CUDA
if available for speed.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import torch

import lib
import lib.graph.data
from bin.graphpfn.pretrain import evaluate_dataset, get_synthetic_eval_dataset
from lib.graph.data import GraphDataset
from lib.graphpfn.model import GraphPFN

DEFAULT_TOML_PATH = Path("exp/graphpfn/pretrain/graph_level/pretrain.toml")
DEFAULT_OUTPUT_DIR = Path("exp/graphpfn/pretrain/graph_level/pretrain")


def evaluate_dataset_no_relabel_standardization(
    graphpfn: torch.nn.Module,
    dataset: GraphDataset,
    device: torch.device,
    amp_enabled: bool,
    parts: list[str] = ["test"],
) -> dict[str, Any]:
    """Copy of `evaluate_dataset` with the label re-standardization removed.

    graph_level labels are already population-standardized (see module
    docstring), so this version uses them as-is: no `prepare_labels` call,
    no inverse pred_transform. Everything else (feature_fit_mask handling,
    masks, model call) is identical to the real `evaluate_dataset`.
    """
    assert dataset.task.is_transductive
    assert dataset.task.is_regression, "this script only targets the graph_level (regression) prior"

    dataset = dataset.to_torch(device)
    features = lib.graph.data.flatten_features(dataset.features)
    assert features is not None
    feature_fit_mask = dataset.data.get("feature_fit_mask")
    if feature_fit_mask is None:
        feature_fit_mask = dataset.data["masks"]["train"]
    features = lib.graph.data.drop_constant_features(features, feature_fit_mask)  # type: ignore
    y_train = dataset.data["labels"][dataset.data["masks"]["train"]].to(  # type: ignore
        dtype=torch.float32, device=device
    )

    with torch.autocast(
        device.type,
        enabled=amp_enabled,
        dtype=torch.bfloat16 if amp_enabled else None,
    ):
        out = graphpfn(
            graph=dataset.data["graph"],
            features=features,
            y_train=y_train,
            train_mask=dataset.data["masks"]["train"],
            task_type=dataset.task.type_,
            n_random_features=8,
        )

    predictions: dict[str, np.ndarray] = {}
    for part in parts:
        predictions[part] = (
            out["predictions"][dataset.data["masks"][part], ...].cpu().numpy()
        )
    # No pred_transform: predictions are already on the same (population-
    # standardized) scale as dataset.task.labels, which was never mutated.

    for part in list(predictions.keys()):
        if predictions[part].shape[0] == 0:
            predictions.pop(part)

    return (
        dataset.task.calculate_metrics(predictions, "labels")
        if lib.are_valid_predictions(predictions)
        else {x: {"score": -999999.0} for x in predictions}
    )


def load_ema_model(config: dict, checkpoint: dict, device: torch.device) -> torch.nn.Module:
    """Loads the EMA weights -- what real periodic eval (`eval_fn(graphpfn_ema)`)
    actually scores against, as opposed to the live (non-EMA) training weights.
    """
    graphpfn = GraphPFN(**config.get("model", dict())).to(device)
    state_dict = {
        k[len("module."):]: v
        for k, v in checkpoint["model_ema"].items()
        if k.startswith("module.")
    }
    missing, unexpected = graphpfn.load_state_dict(state_dict, strict=False)
    assert not missing, f"missing keys when loading EMA weights: {missing}"
    assert not unexpected, f"unexpected keys when loading EMA weights: {unexpected}"
    graphpfn.eval()
    return graphpfn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toml", type=Path, default=DEFAULT_TOML_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-datasets", type=int, default=50)
    args = parser.parse_args()

    with open(args.toml, "rb") as f:
        toml_config = tomllib.load(f)
    config = toml_config["base_config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = lib.load_checkpoint(args.output_dir, map_location=device)
    print(f"Loaded checkpoint from {args.output_dir} at step={checkpoint['step']}")

    amp_enabled = (
        config.get("amp", False)
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    print(f"device={device} amp_enabled={amp_enabled}")

    graphpfn = load_ema_model(config, checkpoint, device)

    buggy_scores: list[float] = []
    fixed_scores: list[float] = []
    n_degenerate = 0

    print(
        f"\n{'idx':>4} {'n_nodes':>8} {'n_train':>8} {'degenerate':>11} "
        f"{'buggy_R2':>12} {'fixed_R2':>12} {'delta':>10}"
    )
    with torch.inference_mode():
        for idx in range(args.n_datasets):
            # Two independent (but identical, since generation is
            # deterministic in idx/config/seed) draws: `evaluate_dataset`
            # mutates dataset.data["labels"] in place via prepare_labels,
            # so reusing one object across both eval calls would leak the
            # buggy run's re-standardization into the "fixed" run.
            dataset_for_buggy = get_synthetic_eval_dataset(idx, config["prior"], config["seed"])
            dataset_for_fixed = get_synthetic_eval_dataset(idx, config["prior"], config["seed"])

            n_nodes = dataset_for_buggy.data["labels"].shape[0]
            n_train = int(dataset_for_buggy.data["masks"]["train"].sum())
            is_degenerate = n_train <= 1
            if is_degenerate:
                n_degenerate += 1

            buggy_metrics = evaluate_dataset(
                graphpfn, dataset_for_buggy, device=device, amp_enabled=amp_enabled, parts=["test"]
            )
            fixed_metrics = evaluate_dataset_no_relabel_standardization(
                graphpfn, dataset_for_fixed, device=device, amp_enabled=amp_enabled, parts=["test"]
            )

            buggy_score = buggy_metrics["test"]["score"]
            fixed_score = fixed_metrics["test"]["score"]
            buggy_scores.append(buggy_score)
            fixed_scores.append(fixed_score)

            print(
                f"{idx:>4} {n_nodes:>8} {n_train:>8} {str(is_degenerate):>11} "
                f"{buggy_score:>12.4f} {fixed_score:>12.4f} {fixed_score - buggy_score:>10.4f}"
            )

    buggy_mean = float(np.mean(buggy_scores))
    fixed_mean = float(np.mean(fixed_scores))
    # Also compute means with degenerate (-999999-poisoned) draws excluded,
    # since a single degenerate draw dominates and hides everything else.
    buggy_arr = np.array(buggy_scores)
    fixed_arr = np.array(fixed_scores)
    non_degenerate = buggy_arr > -1000.0  # -999999 sentinel for invalid predictions
    buggy_mean_clean = float(buggy_arr[non_degenerate].mean()) if non_degenerate.any() else float("nan")
    fixed_mean_clean = float(fixed_arr[non_degenerate].mean()) if non_degenerate.any() else float("nan")

    print(f"\n{'=' * 70}")
    print(f"n_datasets={args.n_datasets}  n_degenerate(n_train<=1)={n_degenerate}")
    print(f"mean R2 (buggy, current code)          = {buggy_mean:.4f}")
    print(f"mean R2 (fixed, no re-standardization)  = {fixed_mean:.4f}")
    print(f"mean R2 (buggy, excl. degenerate draws) = {buggy_mean_clean:.4f}")
    print(f"mean R2 (fixed, excl. degenerate draws) = {fixed_mean_clean:.4f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
