"""Graph-level ("whole-molecule") pretraining entrypoint: trains the
learnable-pooling adapter (`GraphLevelGraphPFN`, see lib/graphpfn/pooling.py)
on the synthetic multi-graph molecule prior, with periodic zero-shot/ICL
evaluation on real QM9 molecules interleaved into normal training (no
gradient from QM9 -- the same "train on prior, eval on real chemistry" check
`dev/pool_icl_real_limix.py` already does, just running periodically during
training instead of once at the end).

This is a parallel vertical to `pretrain.py`, not a modification of it: the
main pipeline's `TaskType`/`GraphTask`/`evaluate_dataset` all assume
node-level transductive semantics (see the plan), so graph-level tasks are
built as their own entrypoint, reusing `pretrain.py`'s DDP/EMA/checkpointing/
gradient-accumulation scaffolding (genuinely task-agnostic) but with a fully
separate step/eval function and data-sampling path.

DDP note: each rank samples its own multi-graph molecule batch independently
(own RNG stream, seeded by rank) rather than a master-samples-then-scatter
protocol -- `GraphPriorSamplerDDP`'s padded fixed-shape scatter design
doesn't fit variable-size `dgl.batch` graphs anyway (approved simplification,
see the plan).
"""

import json
import math
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, NotRequired

import delu
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import r2_score
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from typing_extensions import TypedDict

import lib
from lib import KWArgs
from lib.graphpfn.model import GraphPFN
from lib.graphpfn.pooling import GraphLevelGraphPFN
from lib.graphpfn.prior.graph_level import (
    GraphLevelPriorSampler,
    raw_sample_to_graph,
    sample_graph_level_dataset,
)
from lib.graphpfn.qm9_data import QM9_TARGETS, load_qm9, sample_one_qm9_dataset
from lib.util import backup, make_seed, tracker


class QM9EvalConfig(TypedDict):
    root: str
    target: str
    n_graphs_min: int
    n_graphs_max: int
    train_ratio: float
    n_eval_datasets: int


class Config(TypedDict):
    seed: int
    amp: NotRequired[bool]
    tracker: NotRequired[KWArgs]

    base_checkpoint: NotRequired[str | Path]
    n_steps: int
    n_gradient_accumulation_steps: int
    lr_scheduler: NotRequired[KWArgs]
    ema: NotRequired[KWArgs]
    gradient_clipping_norm: NotRequired[float]
    optimizer: KWArgs
    epoch_size: int

    prior: KWArgs  # base_prior_config, as passed to sample_graph_level_dataset
    model: NotRequired[KWArgs]  # GraphPFN kwargs
    pooling: NotRequired[KWArgs]  # GraphLevelGraphPFN kwargs (n_geometric_rounds, etc.)
    sampler: NotRequired[KWArgs]  # GraphLevelPriorSampler kwargs (n_workers, prefetch_factor)

    n_synthetic_eval_datasets: int
    qm9_eval: NotRequired[QM9EvalConfig]


def context_query_split(
    n_graphs: int, train_ratio: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    n_train = max(1, min(n_graphs - 1, round(n_graphs * train_ratio)))
    perm = rng.permutation(n_graphs)
    return perm[:n_train], perm[n_train:]


def standardize_by_context(y_all: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Standardize using ALL molecules (context + query), matching the
    dev script's own convention -- not context-only stats, since with
    context sizes as small as a couple of molecules, a context-only std
    estimate can be near-zero and blow up the standardized target. See
    `pool_icl_real_limix.py`'s `standardize_by_context` for the full
    rationale (this caused every extreme loss spike seen in that series).
    """
    mean = y_all.mean()
    std = y_all.std()
    std = std if std > 1e-6 else 1.0
    return (y_all - mean) / std


def _sample_split_dataset_with_retries(
    prior_config: KWArgs,
    rng: np.random.Generator,
    min_n_graphs: int = 4,
    max_attempts: int = 50,
):
    """Samples a dataset AND a valid context/query split together, retrying
    the WHOLE sample (not just the split) on failure.

    A bad (n_graphs, train_ratio) combination (e.g. n_graphs=6,
    train_ratio=0.05 -> round(6*0.05)=0 train molecules, deterministically,
    every time) can make every possible split of that ONE sampled dataset
    invalid -- retrying context_query_split alone on the same dataset would
    loop forever. Resampling the dataset itself (giving a fresh train_ratio)
    is what actually recovers.
    """
    for _ in range(max_attempts):
        try:
            atom_features, graph, y_per_molecule, train_ratio = sample_graph_level_dataset(
                prior_config
            )
        except Exception:
            continue
        n_graphs = int(graph.batch_num_nodes().shape[0])
        if n_graphs < min_n_graphs:
            continue
        train_idx, test_idx = context_query_split(n_graphs, train_ratio, rng)
        y_np = y_per_molecule.numpy()
        if len(test_idx) == 0 or len(train_idx) < 2 or np.std(y_np[test_idx]) == 0:
            continue
        return atom_features, graph, y_np, train_idx, test_idx, n_graphs
    raise RuntimeError(f"Could not sample a valid split dataset after {max_attempts} attempts")


def _next_split_from_sampler(
    sampler: GraphLevelPriorSampler,
    rng: np.random.Generator,
    max_attempts: int = 50,
):
    """Same retry-the-whole-dataset logic as `_sample_split_dataset_with_retries`,
    but pulling already-sampled (or being-prefetched-in-the-background)
    `GraphLevelSample`s from `sampler` instead of calling
    `sample_graph_level_dataset` synchronously.
    """
    for _ in range(max_attempts):
        atom_features, graph, y_per_molecule, train_ratio = raw_sample_to_graph(next(sampler))
        n_graphs = int(graph.batch_num_nodes().shape[0])
        train_idx, test_idx = context_query_split(n_graphs, train_ratio, rng)
        y_np = y_per_molecule.numpy()
        if len(test_idx) == 0 or len(train_idx) < 2 or np.std(y_np[test_idx]) == 0:
            continue
        return atom_features, graph, y_np, train_idx, test_idx, n_graphs
    raise RuntimeError(f"Could not get a valid split sample after {max_attempts} attempts")


def _sample_split_qm9_with_retries(
    qm9_dataset,
    qm9_config: "QM9EvalConfig",
    target_idx: int,
    rng: np.random.Generator,
    max_attempts: int = 50,
):
    for _ in range(max_attempts):
        atom_features, graph, y_per_molecule = sample_one_qm9_dataset(
            qm9_dataset, (qm9_config["n_graphs_min"], qm9_config["n_graphs_max"]), target_idx, rng
        )
        n_graphs = int(graph.batch_num_nodes().shape[0])
        train_idx, test_idx = context_query_split(n_graphs, qm9_config["train_ratio"], rng)
        y_np = y_per_molecule.numpy()
        if len(test_idx) == 0 or len(train_idx) < 2 or np.std(y_np[test_idx]) == 0:
            continue
        return atom_features, graph, y_np, train_idx, test_idx, n_graphs
    raise RuntimeError(f"Could not sample a valid QM9 split after {max_attempts} attempts")


def _forward_one_dataset(
    model: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    atom_features: torch.Tensor,
    graph,
    y_per_molecule: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (query_predictions, query_targets), both on `device`."""
    n_graphs = int(graph.batch_num_nodes().shape[0])
    y_std = standardize_by_context(y_per_molecule, train_idx)

    is_context = torch.zeros(n_graphs, dtype=torch.bool, device=device)
    is_context[torch.from_numpy(train_idx)] = True
    y_std_t = torch.as_tensor(y_std, dtype=torch.float32, device=device)

    atom_features = atom_features.to(device)
    graph = graph.to(device)

    with torch.autocast(
        device.type, enabled=amp_enabled, dtype=torch.bfloat16 if amp_enabled else None
    ):
        pred = model(graph, atom_features, y_std_t, is_context)

    query_idx = torch.from_numpy(test_idx).to(device)
    return pred[query_idx], y_std_t[query_idx]


def main(
    config,
    output: str | Path,
    *,
    force: bool = False,
    continue_: bool = False,
    profiler: torch.profiler.profile | None = None,
    tiny: bool = False,
) -> None | lib.JSONDict:
    warnings.filterwarnings("ignore", category=FutureWarning)

    if lib.is_ddp():
        lib.configure_ddp(timeout_minutes=30)
    lib.configure_logging(enqueue=True)

    logger.info(f"Launching on {torch.cuda.device_count()} devices")

    config, output = lib.check(config, output, config_type=Config)

    if lib.is_master_process():
        start = lib.start(main, output, force=force, continue_=continue_)
        lib.broadcast_bool(start)
    else:
        start = lib.broadcast_bool()
    if not start:
        return None

    output = Path(output)
    rank = lib.get_rank()

    if tiny:
        config["n_steps"] = 6
        config["n_gradient_accumulation_steps"] = 2
        config["epoch_size"] = 2
        config["n_synthetic_eval_datasets"] = 2
        if "qm9_eval" in config:
            config["qm9_eval"]["n_eval_datasets"] = 2

    timer = delu.tools.Timer()
    timer.run()

    loaded_checkpoint = (
        lib.load_checkpoint(output)
        if continue_ and lib.get_checkpoint_path(output).exists()
        else None
    )

    if lib.is_master_process():
        name = str(output.parent.relative_to(lib.env.get_exp_dir()))
        tracker.init(
            config,  # type: ignore
            name,
            run_uid=loaded_checkpoint["run_uid"] if loaded_checkpoint else None,
            step=loaded_checkpoint["step"] + 1 if loaded_checkpoint else 0,
            **config.get("tracker", {}),
        )
        lib.print_config(config)  # type: ignore
        print()

    resume_step = loaded_checkpoint["step"] if loaded_checkpoint is not None else 0
    del loaded_checkpoint

    # Two independent RNG streams, both seeded per-rank so different DDP
    # ranks never sample identical data: `delu.random` for anything the
    # backbone/prior touches through global numpy/random/torch state, and a
    # dedicated np.random.Generator for context/query splitting and QM9
    # sampling (matching dev/pool_icl_real_limix.py's convention).
    delu.random.seed(make_seed(config["seed"], "main", rank, resume_step))
    rng = np.random.default_rng(make_seed(config["seed"], "rng", rank, resume_step))

    # Background-prefetching sampler for the training hot path (step_fn) --
    # n_workers=0 (default) falls back to synchronous sampling, identical to
    # calling sample_graph_level_dataset directly. See GraphLevelPriorSampler
    # (prior/graph_level.py) for why this returns plain tensors rather than a
    # dgl.DGLGraph across the worker-process boundary.
    graph_level_sampler = GraphLevelPriorSampler(
        base_prior_config=config["prior"],
        seed=make_seed(config["seed"], "graph_sampler", rank, resume_step),
        **config.get("sampler", dict()),
    )

    step = 0
    device = lib.get_device()
    report = lib.create_report(main, config)  # type: ignore

    # >>> model
    # edge_head has no Python default (GraphPFN.__init__ requires it
    # explicitly) but TOML has no way to express None -- default to no edge
    # reconstruction head unless the config opts in with a
    # [base_config.model.edge_head] table.
    model_kwargs = dict(config.get("model", dict()))
    model_kwargs.setdefault("edge_head", None)
    graphpfn = GraphPFN(**model_kwargs).to(device)

    if "base_checkpoint" in config:
        checkpoint = torch.load(config["base_checkpoint"], map_location="cpu")
        if "model_ema" in checkpoint:
            state_dict = {
                k[7:]: v for k, v in checkpoint["model_ema"].items() if k.startswith("module.")
            }
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            raise ValueError(
                "Unrecognized base_checkpoint format: expected a 'model_ema' or "
                f"'state_dict' key, got top-level keys={list(checkpoint.keys())}"
            )
        assert all(k in graphpfn.state_dict() for k in state_dict.keys())
        graphpfn.load_state_dict(state_dict, strict=False)

    model = GraphLevelGraphPFN(
        graphpfn, embed_dim=graphpfn.tfm.module.embed_dim, **config.get("pooling", dict())
    ).to(device)

    report["n_parameters"] = lib.deep.get_n_parameters(model)
    logger.info(f"n_parameters = {report['n_parameters']}")

    model_without_ddp = model
    if lib.is_ddp():
        model = DistributedDataParallel(
            model, device_ids=[rank], output_device=rank, find_unused_parameters=True
        )

    if "ema" in config:
        multi_avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(decay=config["ema"]["decay"])
        model_ema = torch.optim.swa_utils.AveragedModel(
            model_without_ddp, device, multi_avg_fn=multi_avg_fn
        )
    else:
        model_ema = model_without_ddp

    # >>> QM9 (real-data periodic eval only -- never trained on)
    qm9_dataset = None
    qm9_config = config.get("qm9_eval")
    qm9_target_idx = None
    if qm9_config is not None:
        qm9_dataset = load_qm9(qm9_config["root"])
        qm9_target_idx = QM9_TARGETS.index(qm9_config["target"])
        logger.info(
            f"QM9 periodic eval enabled: target={qm9_config['target']!r}, "
            f"{len(qm9_dataset)} molecules loaded"
        )

    # >>> prepare training
    params = lib.deep.make_parameter_groups(model_without_ddp)
    optimizer = lib.deep.make_optimizer(**config["optimizer"], params=params)
    gradient_clipping_norm = config.get("gradient_clipping_norm")
    epoch_size = config["epoch_size"]

    if "lr_scheduler" in config:
        lr_scheduler = lib.deep.get_lr_scheduler(
            optimizer=optimizer, n_steps=config["n_steps"], **config["lr_scheduler"]
        )
    else:
        lr_scheduler = None

    amp_enabled = (
        config.get("amp", False) and device.type == "cuda" and torch.cuda.is_bf16_supported()
    )
    logger.info(f"AMP enabled: {amp_enabled}")

    def prepare_checkpoint(step: int) -> dict:
        return {
            "run_uid": tracker.get_uid(),
            "step": step,
            "report": report,
            "timer": timer,
            "model": model_without_ddp.state_dict(),
            "model_ema": model_ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            "random_state": delu.random.get_state(),
            "extra": {},
        }

    def save_checkpoint(checkpoint: dict) -> None:
        lib.barrier()
        if lib.is_master_process():
            info = {k: v["score"] for k, v in checkpoint["report"]["metrics"].items()} | {
                "lr": checkpoint["report"]["lr"][0] if "lr" in checkpoint["report"] else None,
                "loss": checkpoint["report"].get("loss"),
            }
            logger.info(f"{info=}")
            tracker.log(info, step=checkpoint["step"])
            with open(output / "training_log.jsonl", "a") as f:
                f.write(json.dumps({"step": checkpoint["step"], **info}) + "\n")
            lib.dump_checkpoint(output, checkpoint)
            backup(output)
        lib.barrier()

    def load_from_checkpoint(checkpoint: dict) -> tuple[int, dict, delu.tools.Timer]:
        model_without_ddp.load_state_dict(checkpoint["model"])
        model_ema.load_state_dict(checkpoint["model_ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if lr_scheduler is not None:
            assert checkpoint["lr_scheduler"] is not None
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        logger.info(
            f"Continuing from step={checkpoint['step']} for run_uid={checkpoint['run_uid']}"
        )
        return checkpoint["step"], checkpoint["report"], checkpoint["timer"]

    # >>> eval

    @torch.inference_mode()
    def eval_fn(eval_model: nn.Module) -> dict[str, dict[str, float]]:
        eval_model.eval()
        metrics: dict[str, dict[str, float]] = {}

        synthetic_scores = []
        for _ in range(config["n_synthetic_eval_datasets"]):
            atom_features, graph, y_np, train_idx, test_idx, _ = (
                _sample_split_dataset_with_retries(config["prior"], rng)
            )
            pred, target = _forward_one_dataset(
                eval_model, device, amp_enabled, atom_features, graph, y_np, train_idx, test_idx
            )
            synthetic_scores.append(r2_score(target.cpu().numpy(), pred.float().cpu().numpy()))
        if synthetic_scores:
            metrics["synthetic"] = {
                "score": lib.allreduce_float(float(np.mean(synthetic_scores)))
            }

        if qm9_dataset is not None:
            qm9_scores = []
            for _ in range(qm9_config["n_eval_datasets"]):
                atom_features, graph, y_np, train_idx, test_idx, _ = (
                    _sample_split_qm9_with_retries(qm9_dataset, qm9_config, qm9_target_idx, rng)
                )
                pred, target = _forward_one_dataset(
                    eval_model, device, amp_enabled, atom_features, graph, y_np, train_idx, test_idx
                )
                qm9_scores.append(r2_score(target.cpu().numpy(), pred.float().cpu().numpy()))
            if qm9_scores:
                metrics["qm9"] = {"score": lib.allreduce_float(float(np.mean(qm9_scores)))}

        eval_model.train()
        return metrics

    # >>> step function

    def step_fn() -> torch.Tensor:
        atom_features, graph, y_np, train_idx, test_idx, _ = (
            _next_split_from_sampler(graph_level_sampler, rng)
        )
        pred, target = _forward_one_dataset(
            model, device, amp_enabled, atom_features, graph, y_np, train_idx, test_idx
        )
        return F.mse_loss(pred, target)

    # >>> training

    report["train"] = {"metrics": {"synthetic": {"score": -math.inf}}}
    n_steps = config["n_steps"]

    lib.barrier()
    checkpoint = (
        lib.load_checkpoint(output)
        if continue_ and lib.get_checkpoint_path(output).exists()
        else None
    )
    is_resuming = checkpoint is not None
    lib.barrier()
    if checkpoint is not None:
        step, report, timer = load_from_checkpoint(checkpoint)
        del checkpoint

    timer.run()

    if not is_resuming:
        delu.cuda.free_memory()
        metrics = eval_fn(model_ema)
        report["train"]["best_step"] = step  # type: ignore
        report["train"]["metrics"] = metrics
        report["metrics"] = metrics
        report["lr"] = (
            lr_scheduler.get_last_lr() if lr_scheduler is not None else config["optimizer"]["lr"]
        )
        report["loss"] = float("nan")
        save_checkpoint(prepare_checkpoint(step))

    while n_steps == -1 or step < n_steps:
        logger.info(f"[...] {output} | {timer}")

        model.train()
        epoch_losses = []
        delu.cuda.free_memory()

        iterator = range(epoch_size)
        if lib.is_master_process():
            iterator = tqdm(iterator, desc=f"Epoch {step // epoch_size} Step {step}")

        for _ in iterator:
            step_loss = []
            for inner_step in range(config["n_gradient_accumulation_steps"]):
                is_gradient_step = (inner_step + 1) == config["n_gradient_accumulation_steps"]

                with (
                    model.no_sync()
                    if lib.is_ddp() and not is_gradient_step
                    else nullcontext()
                ):
                    loss = step_fn()
                    if not torch.isfinite(loss):
                        logger.warning(
                            f"[step {step}] non-finite loss ({loss.item()}), skipping micro-step"
                        )
                        continue
                    (loss / config["n_gradient_accumulation_steps"]).backward()
                    step_loss.append(loss.detach())

                if profiler is not None:
                    profiler.step()

            if not step_loss:
                # every micro-step this round was non-finite; drop the whole
                # accumulated step without touching the optimizer, same
                # rationale as pool_icl_real_limix.py's guard -- a bad step
                # must never reach optimizer.step().
                optimizer.zero_grad()
                step += 1
                continue

            if gradient_clipping_norm is not None:
                grad_norm = nn.utils.clip_grad.clip_grad_norm_(
                    model.parameters(), gradient_clipping_norm
                )
                if not torch.isfinite(grad_norm):
                    logger.warning(f"[step {step}] non-finite grad norm, skipping step")
                    optimizer.zero_grad()
                    step += 1
                    continue

            optimizer.step()
            if isinstance(model_ema, torch.optim.swa_utils.AveragedModel):
                model_ema.update_parameters(model_without_ddp)
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad()

            step += 1
            epoch_losses.append(torch.mean(torch.stack(step_loss)))

        epoch_loss_mean = (
            torch.stack(epoch_losses).mean() if epoch_losses else torch.tensor(float("nan"))
        )
        if lib.is_ddp():
            torch.distributed.all_reduce(epoch_loss_mean)
            epoch_loss_mean = epoch_loss_mean / lib.get_world_size()

        delu.cuda.free_memory()
        metrics = eval_fn(model_ema)

        report["train"]["best_step"] = step  # type: ignore
        report["train"]["metrics"] = metrics
        report["metrics"] = metrics
        report["lr"] = (
            lr_scheduler.get_last_lr() if lr_scheduler is not None else config["optimizer"]["lr"]
        )
        report["loss"] = epoch_loss_mean.item()

        save_checkpoint(prepare_checkpoint(step))

    # >>> finish
    lib.barrier()
    report["time"] = timer.elapsed()
    if lib.is_master_process():
        lib.finish(output, report)
    if lib.is_ddp():
        torch.distributed.destroy_process_group()
    return report


if __name__ == "__main__":
    lib.configure_torch(deterministic=False)
    lib.run(main)
