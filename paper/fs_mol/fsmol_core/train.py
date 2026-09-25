"""Episodic meta-training of the pooler directly on FS-Mol's real Dtrain assays, for any recipe.

Only the pooler trains -- the frozen LimiX-16M backbone keeps requires_grad=False throughout. Each
optimizer step accumulates gradients over --grad-accum-steps EPISODES; each episode:
  1. samples one task uniformly from Dtrain (fs-mol/train/*.jsonl.gz, "train" list of fsmol-0.1.json),
     skipping degenerate tasks (<4 molecules or single-class) permanently;
  2. loads / featurizes it ONCE and caches it in memory for the life of the run (recipes.featurize);
  3. splits it into a stratified support (context) / query set (data.stratified_support_query, the
     byte-identical FS-Mol protocol), support size drawn per episode from --support-sizes, query capped
     to --max-query molecules;
  4. runs the same forward pass as the evaluator but WITH gradients and backprops
     F.cross_entropy(pred, target) on the query predictions.
AdamW + cosine/warmup LR, gradient clipping, EMA of the pooler weights, bf16 autocast, crash-safe
checkpoint save/resume (schema {"step","pooler","pooler_ema","optimizer","lr_scheduler",
"best_held_out_ap","best_step"} -- the evaluators load it unmodified).

Checkpoint selection ("held-out AP" in the schema, but actually mean delta-AUPRC here): every
--eval-every steps run the fixed FS-Mol in-context evaluation (EMA weights, no grad) on the 40 Dvalid
tasks at --valid-support-sizes with --valid-num-runs resamples each; the best value at
--valid-selection-size is saved as pooler_checkpoint_best.pt.

MULTI-GPU (DDP): plain `python` runs single-process; under `python -m torch.distributed.run
--nproc_per_node=N` (lib.is_ddp()) every rank samples its OWN episodes (rank-offset seed) and
gradients are averaged manually with one all_reduce per optimizer step (no DistributedDataParallel
wrapper: the forward runs through the raw module). Dvalid evaluation, checkpointing and logging are
rank 0 only; its early-stop decision is broadcast so all ranks stay in lockstep. An existing
pooler_checkpoint.pt in --output-dir is resumed automatically; otherwise training starts from
--pooler-checkpoint (default: bt.WARMSTART_CHECKPOINT_PATH).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import cli, data, forward, metrics, recipes
from .recipes import Recipe

DEFAULT_SUPPORT_SIZES = "16,32,64,128"
DEFAULT_VALID_SUPPORT_SIZES = "16,128"


def parse_args(recipe: Recipe, description: str, output_dir: Path) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_data_args(p, "train/valid/")
    p.add_argument("--pooler-checkpoint", type=Path, default=None,
                   help="Starting pooler checkpoint (default: bt.WARMSTART_CHECKPOINT_PATH, the pretrained-as-a-PFN pooler).")
    p.add_argument("--device", default=None, help="Ignored under torchrun/DDP (each rank uses its own GPU automatically); applies only to a plain single-process run.")
    p.add_argument("--lr", type=float, default=1e-4, help="Peak LR, cosine+warmup (default: 1e-4 -- lower than the pretrain default 1e-3 since this continues an already-trained pooler).")
    p.add_argument("--n-steps", type=int, default=20000, help="Total optimizer steps.")
    p.add_argument("--grad-accum-steps", type=int, default=20, help="Episodes accumulated per optimizer step.")
    p.add_argument("--support-sizes", type=cli.parse_int_list, default=None,
                   help=f"Comma list of support sizes to sample episodes at (default: {DEFAULT_SUPPORT_SIZES}).")
    p.add_argument("--max-query", type=int, default=128, help="Cap each episode's query set to N molecules (compute/memory).")
    if recipe.splice:
        p.add_argument("--fp-fold", type=int, default=0,
                       help="Fold the 2048-dim ECFP fingerprint splice to this width (classic modulo folding, same as the "
                            "evaluator's --fp-fold; <=0 or >=2048 = no folding, i.e. raw/unfolded). Default 0 (raw). Lower "
                            "widths cut the per-molecule token count the splice adds to the transformer -- use this if training OOMs.")
    cli.add_edge_arg(p)
    p.add_argument("--eval-every", type=int, default=200, help="Steps between Dvalid evaluations / checkpoint saves.")
    p.add_argument("--valid-support-sizes", type=cli.parse_int_list, default=None,
                   help=f"Comma list of support sizes for Dvalid evaluation (default: {DEFAULT_VALID_SUPPORT_SIZES}).")
    p.add_argument("--valid-num-runs", type=int, default=5, help="Resamples per (valid task, support size).")
    p.add_argument("--valid-selection-size", type=int, default=16,
                   help="Which of --valid-support-sizes' mean delta-AUPRC drives checkpoint selection (default: 16, matching the paper's Table 2).")
    p.add_argument("--patience", type=int, default=25, help="Consecutive evals without Dvalid improvement before stopping early (like fs_mol's own train_loop).")
    p.add_argument("--limit-train-tasks", type=int, default=None, help="First N Dtrain tasks only (debugging).")
    p.add_argument("--limit-valid-tasks", type=int, default=None, help="First N Dvalid tasks only (debugging).")
    p.add_argument("--seed", type=int, default=0)
    fold_tag = "_fp_<fold>" if recipe.splice else ""
    p.add_argument("--output-dir", type=Path, default=None, help=f"Default: {output_dir}/lr_<lr>_supp_<sizes>{fold_tag}.")
    return p.parse_args()


# --------------------------------------------------------------------------------------------------
# task cache / episode sampling (training) / fixed-protocol evaluation (Dvalid)
# --------------------------------------------------------------------------------------------------

class TaskCache:
    """Lazily loads (and featurizes) FS-Mol tasks from `data_dir`, caching successes in memory and
    remembering degenerate / unbuildable tasks so they are never retried."""

    def __init__(self, recipe: Recipe, env, data_dir: Path, fp_fold: int):
        self.recipe, self.env, self.data_dir, self.fp_fold = recipe, env, data_dir, fp_fold
        self._cache: dict[str, data.TaskData] = {}
        self._bad: dict[str, str] = {}

    def get(self, name: str) -> data.TaskData | None:
        if name in self._bad:
            return None
        if name in self._cache:
            return self._cache[name]
        task = recipes.read_task(self.recipe, self.data_dir / f"{name}.jsonl.gz", fp_fold=self.fp_fold)
        if task.degenerate:
            self._bad[name] = "degenerate"
            return None
        try:
            recipes.featurize(self.recipe, self.env, task, name)
        except recipes.UnbuildableTask as e:
            self._bad[name] = str(e)
            return None
        self._cache[name] = task
        return task

    @property
    def n_bad(self) -> int:
        return len(self._bad)


def sample_training_episode(cache: TaskCache, task_names: list[str], support_sizes: list[int],
                            max_query: int, fpg: int, bidirectional: bool, rng, max_attempts: int = 100):
    """Samples one (task, support_size) pair, retrying on degenerate tasks / too-small splits, and
    returns (raw, permol_ordered, task_name, support_size) -- or None if max_attempts is exhausted
    (the caller just skips this micro-step; only happens if most of Dtrain is degenerate)."""
    for _ in range(max_attempts):
        name = task_names[rng.integers(len(task_names))]
        task = cache.get(name)
        if task is None:
            continue

        support_size = int(support_sizes[rng.integers(len(support_sizes))])
        seed = int(rng.integers(2**31 - 1))
        try:
            sup_idx, qry_idx = data.stratified_support_query(task.labels, support_size, seed)
        except data.SplitTooSmall:
            continue
        qry_idx = data.cap_query(qry_idx, max_query, rng)
        raw, permol = data.make_episode(task, sup_idx, qry_idx, fpg, bidirectional)
        return raw, permol, name, support_size
    return None


def evaluate_on_tasks(env, model, pooler_module, device, cache: TaskCache, task_names: list[str],
                      support_sizes: list[int], num_runs: int, fpg: int, bidirectional: bool,
                      base_seed: int) -> dict[int, dict]:
    """Fixed FS-Mol protocol (num_runs stratified resamples per task and support size), EMA weights,
    no grad -- the same in-context evaluation the evaluators run on Dtest. Returns
    {support_size: {"delta_auprc": mean, "n_tasks": n, "per_task": {name: mean_delta_auprc}}}."""
    import torch

    pooler_module.eval()
    out: dict[int, dict] = {}
    for size in support_sizes:
        per_task: dict[str, float] = {}
        for name in task_names:
            task = cache.get(name)
            if task is None:
                continue
            run_scores = []
            for run_idx in range(num_runs):
                try:
                    sup_idx, qry_idx = data.stratified_support_query(task.labels, size, base_seed + run_idx)
                except data.SplitTooSmall:
                    continue
                raw, permol = data.make_episode(task, sup_idx, qry_idx, fpg, bidirectional)
                with torch.no_grad(), env.bt._autocast_ctx(device):
                    dataset = forward.encode_dataset(env, model, device, raw)
                    grouped = None if permol is None else forward.splice_groups(env, model, device, permol, raw)
                    pos_prob, y_true = forward.predict(env, model, dataset, pooler_module, grouped)
                run_scores.append(metrics.point_metrics(pos_prob, y_true)["delta_auprc"])
            if run_scores:
                per_task[name] = float(np.mean(run_scores))
        mean_delta_auprc = float(np.mean(list(per_task.values()))) if per_task else float("nan")
        out[size] = {"delta_auprc": mean_delta_auprc, "n_tasks": len(per_task), "per_task": per_task}
    return out


# --------------------------------------------------------------------------------------------------

def main(recipe: Recipe, description: str, output_dir: Path) -> None:
    args = parse_args(recipe, description, output_dir)
    env = recipes.bootstrap(recipe, args.paper_dir)
    import lib  # noqa: E402 -- importable once paper_dir is on sys.path (pooler_glue.load did that).
    import lib.deep  # noqa: E402

    import torch
    import torch.nn.functional as F

    bt = env.bt
    fp_fold = getattr(args, "fp_fold", 0)

    # DDP: the launcher sets RANK/WORLD_SIZE/LOCAL_RANK in the environment -- lib.is_ddp() is False
    # (and every rank/world_size/is_main default below is a plain single-process run) whenever this
    # script is launched with plain `python` instead of `python -m torch.distributed.run
    # --nproc_per_node=N`. --device only applies in that single-process fallback; under DDP each rank
    # always gets its own GPU via lib.configure_ddp()/lib.get_device().
    if lib.is_ddp():
        lib.configure_ddp(timeout_minutes=60)
    rank = lib.get_rank()
    world_size = lib.get_world_size()
    is_main = lib.is_master_process()
    device = torch.device(args.device) if (args.device and not lib.is_ddp()) else lib.get_device()

    support_sizes = args.support_sizes or cli.parse_int_list(DEFAULT_SUPPORT_SIZES)
    valid_support_sizes = args.valid_support_sizes or cli.parse_int_list(DEFAULT_VALID_SUPPORT_SIZES)
    bidirectional = not args.keep_edge_direction
    # Rank-offset seeding so DDP workers sample DIFFERENT FS-Mol episodes each step.
    rng = np.random.default_rng(args.seed + rank * 100_000)
    torch.manual_seed(args.seed + rank)

    n_cols = recipes.n_atom_cols(recipe, env)
    if is_main:
        print(f"DDP: world_size={world_size}" if lib.is_ddp() else "Single-process run (no torchrun)")
        print(f"atom features: {recipe.atoms_desc} ({n_cols} cols, pre features_per_group padding)")
        if recipe.splice:
            print("per-molecule splice: fingerprint"
                  + (" (RAW 2048-dim ECFP, unfolded)" if fp_fold <= 0 else f" (folded to {fp_fold})"))

    fsmol_dir = args.fsmol_dir
    data_dir = args.fsmol_data or fsmol_dir
    task_list_json = args.task_list_json or (fsmol_dir / "fsmol-0.1.json")
    task_lists = data.load_task_lists(task_list_json)
    train_task_names = sorted(task_lists["train"])
    valid_task_names = sorted(task_lists["valid"])
    if args.limit_train_tasks is not None:
        train_task_names = train_task_names[: args.limit_train_tasks]
    if args.limit_valid_tasks is not None:
        valid_task_names = valid_task_names[: args.limit_valid_tasks]

    run_tag = f"lr_{args.lr:g}_supp_{'-'.join(map(str, support_sizes))}"
    if recipe.splice:
        run_tag += f"_fp_{'raw' if fp_fold <= 0 else fp_fold}"
    run_dir = args.output_dir or (output_dir / run_tag)
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
    lib.barrier()
    checkpoint_path = run_dir / "pooler_checkpoint.pt"
    best_checkpoint_path = run_dir / "pooler_checkpoint_best.pt"
    log_path = run_dir / "training_log.jsonl"

    if is_main:
        print(f"Dtrain tasks: {len(train_task_names)}  |  Dvalid tasks: {len(valid_task_names)}")
        print(f"episode support sizes: {support_sizes}  |  max_query: {args.max_query}  |  grad_accum: {args.grad_accum_steps}  |  world_size: {world_size}")
        print(f"Dvalid eval support sizes: {valid_support_sizes}  |  num_runs: {args.valid_num_runs}  |  selection size: {args.valid_selection_size}")
        print(f"Output dir: {run_dir}")

    model = forward.load_frozen_backbone(env, device)
    fpg = model.features_per_group
    if is_main:
        print(f"  features_per_group={fpg}  ({n_cols}-col atom features padded to {n_cols + (-n_cols % fpg)})")

    # pooler_without_ddp is the module whose state_dict is checkpointed AND the module every
    # forward/backward pass runs through (no DistributedDataParallel wrapper -- see module docstring).
    pooler_without_ddp = bt.PoolingGNN(
        embed_dim=model.embed_dim, n_layers=bt.N_GNN_LAYERS, dropout=bt.DROPOUT,
        n_transformer_blocks=model.nlayers, n_heads=bt.N_ATTN_HEADS,
    ).to(device)

    ema_multi_avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(decay=bt.EMA_DECAY)
    pooler_ema = torch.optim.swa_utils.AveragedModel(pooler_without_ddp, device, multi_avg_fn=ema_multi_avg_fn)

    params = lib.deep.make_parameter_groups(pooler_without_ddp)
    optimizer = lib.deep.make_optimizer(type=bt.OPTIMIZER_TYPE, lr=args.lr, weight_decay=bt.WEIGHT_DECAY, params=params)
    n_warmup_steps = max(1, round(args.n_steps * bt.WARMUP_FRACTION))
    lr_scheduler = lib.deep.get_lr_scheduler(optimizer, n_warmup_steps=n_warmup_steps, n_steps=args.n_steps, scheduler=bt.LR_SCHEDULER)

    start_step = 1
    best_delta_auprc = float("-inf")
    best_step = -1
    evals_since_best = 0
    if checkpoint_path.exists():
        # Every rank reads the same checkpoint file off shared storage independently -- deterministic,
        # no coordination needed (only the SAVE side, further down, is rank-0-only + barriered).
        ckpt = bt.load_checkpoint(checkpoint_path, device)
        pooler_without_ddp.load_state_dict(ckpt["pooler"])
        pooler_ema.load_state_dict(ckpt["pooler_ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        best_delta_auprc = ckpt["best_held_out_ap"]
        best_step = ckpt["best_step"]
        start_step = ckpt["step"] + 1
        if is_main:
            print(f"Resumed from {checkpoint_path} at step {ckpt['step']} (best Dvalid delta-AUPRC so far: {best_delta_auprc:.4f} @ step {best_step})")
    else:
        pooler_checkpoint = args.pooler_checkpoint or bt.WARMSTART_CHECKPOINT_PATH
        if not Path(pooler_checkpoint).exists():
            raise FileNotFoundError(f"starting pooler checkpoint not found: {pooler_checkpoint}")
        src_ckpt = bt.load_checkpoint(pooler_checkpoint, device)
        pooler_without_ddp.load_state_dict(src_ckpt["pooler"], strict=True)
        pooler_ema.load_state_dict(src_ckpt["pooler_ema"])
        if is_main:
            print(f"Starting from {pooler_checkpoint} (source step {src_ckpt.get('step', '?')}). Optimizer + LR schedule start fresh.")

    # Each rank builds its OWN cache (own process memory, own IO/featurization) -- no cross-rank
    # sharing. Dvalid evaluation only ever runs on rank 0 (see below), so only rank 0's valid_cache
    # is ever populated in practice.
    train_cache = TaskCache(recipe, env, data_dir / "train", fp_fold)
    valid_cache = TaskCache(recipe, env, data_dir / "valid", fp_fold)

    def evaluate_valid(base_seed: int) -> dict:
        return evaluate_on_tasks(env, model, pooler_ema.module, device, valid_cache, valid_task_names,
                                 valid_support_sizes, args.valid_num_runs, fpg, bidirectional, base_seed)

    if start_step == 1:
        if is_main:
            print("Evaluating on Dvalid at step 0 (baseline, before any training)...")
            base_eval = evaluate_valid(args.seed)
            for size, r in base_eval.items():
                print(f"  support {size:4d}: delta-AUPRC={r['delta_auprc']:+.4f} over {r['n_tasks']} tasks")
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": 0, "loss": None, **{f"valid_delta_auprc_{s}": r["delta_auprc"] for s, r in base_eval.items()}}) + "\n")
        lib.barrier()

    t0 = time.time()
    for step in range(start_step, args.n_steps + 1):
        pooler_without_ddp.train()
        optimizer.zero_grad()
        step_losses = []
        n_skipped = 0
        for inner_step in range(args.grad_accum_steps):
            sampled = sample_training_episode(train_cache, train_task_names, support_sizes, args.max_query, fpg, bidirectional, rng)
            if sampled is None:
                n_skipped += 1
                continue
            raw, permol, _task_name, _support_size = sampled
            with bt._autocast_ctx(device):
                dataset = forward.encode_dataset(env, model, device, raw)
                # The splice encoding runs under no_grad -- it is a FIXED embedding of the frozen
                # backbone's own encoder, not a trainable path. Gradients still reach the pooler
                # through the OTHER input of the torch.cat inside the forward (`pooled`, which the
                # pooler DOES produce with grad).
                grouped = None if permol is None else forward.splice_groups(env, model, device, permol, raw)
                pred, target = forward.logits(env, model, dataset, pooler_without_ddp, grouped)
                loss = F.cross_entropy(pred, target)
            (loss / args.grad_accum_steps).backward()
            step_losses.append(loss.detach())

        if not step_losses:
            if is_main:
                print(f"step {step:6d} | ALL {args.grad_accum_steps} episodes skipped (degenerate tasks?) -- no update")
            continue

        if lib.is_ddp():
            # Manual gradient averaging across ranks: every rank must reach this same all_reduce for
            # the same parameters in the same order. A rank whose ENTIRE grad_accum_steps was skipped
            # (all episodes degenerate) takes the `continue` above instead and never reaches here,
            # which would hang the others -- accepted, undefended risk (not expected on real Dtrain data).
            for p in pooler_without_ddp.parameters():
                if p.grad is not None:
                    torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
                    p.grad /= world_size

        grad_norm = torch.nn.utils.clip_grad_norm_(pooler_without_ddp.parameters(), bt.GRADIENT_CLIPPING_NORM)
        optimizer.step()
        pooler_ema.update_parameters(pooler_without_ddp)
        lr_scheduler.step()
        step_loss_mean = torch.stack(step_losses).mean().item()

        if is_main:
            elapsed = time.time() - t0
            print(f"step {step:6d} | loss {step_loss_mean:.4f} | grad_norm {grad_norm.item():.4f} | "
                  f"lr {lib.deep.get_lr(optimizer):.2e} | skipped {n_skipped}/{args.grad_accum_steps} | {elapsed:.0f}s")

        should_stop = False
        if step % args.eval_every == 0 or step == args.n_steps:
            if is_main:
                valid_eval = evaluate_valid(args.seed + step)
                for size, r in valid_eval.items():
                    print(f"  Dvalid support {size:4d}: delta-AUPRC={r['delta_auprc']:+.4f} over {r['n_tasks']} tasks")
                selection_metric = valid_eval.get(args.valid_selection_size, {}).get("delta_auprc", float("nan"))

                bt.save_checkpoint(checkpoint_path, step, pooler_without_ddp, pooler_ema, optimizer, lr_scheduler, best_delta_auprc, best_step)
                if selection_metric == selection_metric and selection_metric > best_delta_auprc:  # NaN-safe >
                    best_delta_auprc = selection_metric
                    best_step = step
                    evals_since_best = 0
                    bt.save_checkpoint(best_checkpoint_path, step, pooler_without_ddp, pooler_ema, optimizer, lr_scheduler, best_delta_auprc, best_step)
                    print(f"  new best Dvalid delta-AUPRC@{args.valid_selection_size}={best_delta_auprc:+.4f} @ step {step} -> {best_checkpoint_path}")
                else:
                    evals_since_best += 1
                    print(f"  {evals_since_best} eval(s) since best ({best_delta_auprc:+.4f} @ step {best_step})")

                with open(log_path, "a") as f:
                    f.write(json.dumps({
                        "step": step, "loss": step_loss_mean, "grad_norm": grad_norm.item(),
                        "lr": lib.deep.get_lr(optimizer),
                        **{f"valid_delta_auprc_{s}": r["delta_auprc"] for s, r in valid_eval.items()},
                        "train_bad_tasks": train_cache.n_bad, "valid_bad_tasks": valid_cache.n_bad,
                    }) + "\n")

                should_stop = evals_since_best >= args.patience
                if should_stop:
                    print(f"No Dvalid improvement for {args.patience} evals -- stopping early at step {step}.")
            lib.barrier()
            # Every rank must agree on whether to stop -- only rank 0 evaluates/decides (above), so
            # broadcast its decision; otherwise a rank that kept looping past another rank's early
            # stop would hang forever at the next gradient all_reduce no one else joins.
            should_stop = lib.broadcast_bool(should_stop)
            if should_stop:
                break

    if is_main:
        print(f"\nbest Dvalid delta-AUPRC@{args.valid_selection_size} = {best_delta_auprc:+.4f} at step {best_step}")
        print(f"resume checkpoint: {checkpoint_path}")
        print(f"best checkpoint:   {best_checkpoint_path}")
        print(f"\nScore it on Dtest with:")
        print(f"  python evaluation/{recipe.eval_script} --pooler-checkpoint {best_checkpoint_path}"
              + (f" --fp-fold {fp_fold} --no-baseline" if recipe.splice else ""))
