"""FS-Mol few-shot evaluation of a pooler checkpoint on the 157 held-out test tasks, for any recipe.

Protocol (fs_mol/utils/test_utils.py::eval_model, Sec 5.1 of the FS-Mol paper):
  * support sizes 16/32/64/128/256, --num-runs (10) stratified resamples each, run r seeded seed+r;
  * split = data.stratified_support_query; runs whose split raises are skipped, like eval_model;
  * one in-context forward pass per (task, size, run): examples = support ++ query, n_context = |support|;
  * delta-AUPRC / ROC-AUC / AP, aggregated by metrics.summarise, broken out by EC super-class.

For a recipe with a per-molecule splice, TWO variants are scored in the SAME loop (one dataset encode
per run): the atom features alone (`baseline_variant`) and with the fingerprint splice -- a direct
A/B. --no-baseline skips the first (~halves the compute).

SHARDING (Slurm job array): --shard-index/--num-shards split the task list round-robin
(tasks[shard_index::num_shards]); every worker writes its own '.shard{I}of{N}.json'; combine them
with the recipe's merge script (merge.py).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import cli, data, forward, metrics, recipes
from .recipes import Recipe

DEFAULT_SUPPORT_SIZES = "16,32,64,128,256"
ATOM_FEATURES_TAG = {"native": "fsmol_native_node_features_32col", "x9_plus_extra": "molnet_x9_plus_extra31_40col"}


def parse_args(recipe: Recipe, description: str, script_file: str, output_dir: Path) -> argparse.Namespace:
    stem = Path(script_file).stem
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_data_args(p, "test/")
    p.add_argument("--pooler-checkpoint", type=Path, default=None,
                   help="Pooler checkpoint (default: bt.WARMSTART_CHECKPOINT_PATH).")
    p.add_argument("--no-ema", action="store_true", help="Use raw (non-EMA) pooler weights.")
    p.add_argument("--device", default=None, help="Default: cuda if available else cpu.")
    if recipe.splice:
        p.add_argument("--per-mol-features", choices=tuple(recipes.MODE_TAGS), default="fingerprint",
                       help="Per-molecule vector spliced onto `pooled`. Default: fingerprint (ECFP, the RF baseline feature).")
        p.add_argument("--fp-fold", type=int, default=512,
                       help="Fold the 2048-dim ECFP to this width (classic modulo folding; <=0 or >=2048 = no folding). "
                            "Default 512. Ignored for --per-mol-features descriptors.")
        p.add_argument("--no-baseline", action="store_true",
                       help=f"Skip the {recipe.baseline_variant} (no-splice) forward pass; only score the augmented variant.")
    p.add_argument("--support-sizes", type=cli.parse_int_list, default=None,
                   help=f"Comma list of support sizes (default: {DEFAULT_SUPPORT_SIZES}).")
    p.add_argument("--num-runs", type=int, default=10, help="Resamples per (task, support size). FS-Mol default: 10.")
    p.add_argument("--seed", type=int, default=0, help="Base seed; run r uses seed+r. Default: 0.")
    cli.add_edge_arg(p)
    p.add_argument("--tasks", type=str, default=None,
                   help="Comma-separated task names (ChEMBL ids) to evaluate instead of all 157. NOT paper-comparable.")
    p.add_argument("--limit-tasks", type=int, default=None, help="First N tasks only (NOT paper-comparable).")
    p.add_argument("--max-query", type=int, default=None,
                   help="Cap query set to N mols per run (GPU memory). Default: None = full query (paper).")
    p.add_argument("--shard-index", type=int, default=0,
                   help="This worker's index in [0, num-shards) for a Slurm job array. Default: 0.")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total number of parallel workers; the (post-filter) task list is split round-robin "
                        f"across them. Default: 1 (no sharding). Merge shard outputs with ./{recipe.merge_script}.")
    p.add_argument("--output-json", type=Path, default=None,
                   help=f"Default: {output_dir}/{stem}.json (a '.shard{{I}}of{{N}}.json' suffix is used "
                        "instead when --num-shards > 1).")
    args = p.parse_args()
    if not (0 <= args.shard_index < max(args.num_shards, 1)):
        raise SystemExit(f"--shard-index {args.shard_index} must be in [0, --num-shards={args.num_shards})")
    return args


def select_tasks(args, task_list_json: Path) -> list[str]:
    """Test tasks after --tasks / --limit-tasks / sharding (with the matching warnings)."""
    all_names = sorted(data.load_task_lists(task_list_json)["test"])
    names = all_names
    if args.tasks is not None:
        wanted = [t.strip() for t in args.tasks.split(",") if t.strip()]
        unknown = [t for t in wanted if t not in set(all_names)]
        if unknown:
            raise SystemExit(f"--tasks: not in {task_list_json} 'test' list: {unknown}")
        names = [t for t in all_names if t in set(wanted)]
        print(f"[warn] --tasks: evaluating {len(names)} task(s) {names} -- NOT comparable to the paper's 157-task numbers")
    if args.limit_tasks is not None:
        names = names[: args.limit_tasks]
        print(f"[warn] --limit-tasks {args.limit_tasks}: NOT comparable to the paper's 157-task numbers")
    if args.max_query is not None:
        print(f"[warn] --max-query {args.max_query}: query set is capped, NOT the paper's full-query protocol")
    return names


def score_run(env, recipe: Recipe, model, pooler_module, device, raw: dict, permol, run_baseline: bool,
              aug_variant: str | None) -> dict:
    """{variant: (P(active), y_true)}. The dataset is encoded once; the baseline forward has no splice,
    the augmented forward concatenates the encoded per-molecule matrix onto `pooled`."""
    import torch

    out = {}
    with torch.no_grad(), env.bt._autocast_ctx(device):
        dataset = forward.encode_dataset(env, model, device, raw)
        assert dataset.n_molecules == raw["n_molecules"]
        assert dataset.eval_pos_molecules == raw["eval_pos_molecules"]
        if run_baseline:
            out[recipe.baseline_variant] = forward.predict(env, model, dataset, pooler_module, None)
        if aug_variant is not None:
            grouped = forward.splice_groups(env, model, device, permol, raw)
            out[aug_variant] = forward.predict(env, model, dataset, pooler_module, grouped)
    return out


def main(recipe: Recipe, description: str, script_file: str, output_dir: Path) -> None:
    args = parse_args(recipe, description, script_file, output_dir)
    env = recipes.bootstrap(recipe, args.paper_dir)

    import torch

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_ema = not args.no_ema
    support_sizes = args.support_sizes or cli.parse_int_list(DEFAULT_SUPPORT_SIZES)
    pooler_checkpoint = args.pooler_checkpoint or env.bt.WARMSTART_CHECKPOINT_PATH
    bidirectional = not args.keep_edge_direction

    per_mol_mode = getattr(args, "per_mol_features", "fingerprint")
    fp_fold = getattr(args, "fp_fold", 0)
    run_baseline = (not args.no_baseline) if recipe.splice else True
    variants = recipe.variants(per_mol_mode, run_baseline)
    aug_variant = recipe.augmented_variant(per_mol_mode) if recipe.splice else None

    fsmol_dir = args.fsmol_dir
    data_dir = args.fsmol_data or fsmol_dir
    test_dir = data_dir / "test"
    task_list_json = args.task_list_json or (fsmol_dir / "fsmol-0.1.json")

    test_task_names = select_tasks(args, task_list_json)
    sharded = args.num_shards > 1
    if sharded:
        full_count = len(test_task_names)
        test_task_names = test_task_names[args.shard_index :: args.num_shards]
        print(f"[shard {args.shard_index}/{args.num_shards}] {len(test_task_names)}/{full_count} tasks "
              f"(round-robin tasks[{args.shard_index}::{args.num_shards}]) -- "
              f"combine all shards with {recipe.merge_script} for the paper-comparable number")

    n_cols = recipes.n_atom_cols(recipe, env)
    print(f"FS-Mol test tasks: {len(test_task_names)} (from {task_list_json})")
    print(f"data dir: {data_dir}  |  atom features: {recipe.atoms_desc} ({n_cols} cols)  |  "
          f"edges: {'bidirectional' if bidirectional else 'as-stored'} + isolated-atom self-loops")
    if recipe.splice:
        print(f"per-molecule splice: {per_mol_mode}"
              + (f" (ECFP folded {data.FP_LEN} -> {fp_fold})" if per_mol_mode != "descriptors" and 0 < fp_fold < data.FP_LEN else "")
              + f"  |  variants: {variants}")
    print(f"support sizes: {support_sizes}  |  runs/size: {args.num_runs}")

    ec_by_task = metrics.load_ec_map(fsmol_dir, test_task_names)
    metrics.print_ec_counts(ec_by_task)

    # ---- model + pooler (once) ----------------------------------------------------------------
    print(f"Loading pooler checkpoint from {pooler_checkpoint} (use_ema={use_ema})")
    checkpoint = env.bt.load_checkpoint(pooler_checkpoint, device)
    print(f"  checkpoint step {checkpoint.get('step')}  best_step {checkpoint.get('best_step')}  "
          f"best_held_out_ap {checkpoint.get('best_held_out_ap')}")
    model = forward.load_frozen_backbone(env, device)
    pooler_module = env.ef._load_pooler_module(checkpoint, model, device, use_ema)
    fpg = model.features_per_group
    print(f"  features_per_group={fpg}  ({n_cols}-col atom features padded to {n_cols + (-n_cols % fpg)})")

    # ---- per-task loop --------------------------------------------------------------------
    per_task: dict[str, dict[int, dict[str, dict]]] = {v: {s: {} for s in support_sizes} for v in variants}
    permol_width = None
    repair_status_counts: dict[str, int] = {}
    task_errors: dict[str, str] = {}
    t0 = time.time()

    for ti, name in enumerate(test_task_names):
        task = recipes.read_task(recipe, test_dir / f"{name}.jsonl.gz", per_mol_mode=per_mol_mode, fp_fold=fp_fold)
        if recipe.splice:
            permol_width = task.permol.shape[1]
        print(f"[{ti + 1:3d}/{len(test_task_names)}] {name}: {len(task.graphs)} mols "
              f"({task.n_pos} pos / {len(task.graphs) - task.n_pos} neg)"
              + (f"  permol[{permol_width}]" if recipe.splice else "") + f"  [{time.time() - t0:.0f}s]")
        if task.degenerate:
            print("    skipping (degenerate task)")
            continue
        try:
            recipes.featurize(recipe, env, task, name)
        except recipes.UnbuildableTask as e:
            print(f"    SKIPPING WHOLE TASK -- {recipe.atoms} atom features could not be built for all molecules: {e}")
            task_errors[name] = str(e)
            continue
        if task.repair_status is not None:
            for st in task.repair_status:
                repair_status_counts[st] = repair_status_counts.get(st, 0) + 1
            n_repaired = sum(1 for st in task.repair_status if st != "ok")
            if n_repaired:
                print(f"    {n_repaired}/{len(task.repair_status)} molecules needed SMILES repair (robust_mol) -- all recovered")

        for size in support_sizes:
            run_points: dict[str, list[dict]] = {v: [] for v in variants}
            printed_skip = False
            for run_idx in range(args.num_runs):
                seed = args.seed + run_idx
                try:
                    sup_idx, qry_idx = data.stratified_support_query(task.labels, size, seed)
                except data.SplitTooSmall as e:
                    if not printed_skip:
                        print(f"    support {size:3d}: run skipped ({e})")
                        printed_skip = True
                    continue
                qry_idx = data.cap_query(qry_idx, args.max_query, np.random.default_rng(seed))
                raw, permol = data.make_episode(task, sup_idx, qry_idx, fpg, bidirectional)
                try:
                    res = score_run(env, recipe, model, pooler_module, device, raw, permol, run_baseline, aug_variant)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"    support {size:3d} run {run_idx}: OOM on {len(qry_idx)} query mols "
                          f"-- skipped (use --max-query" + (" / smaller --fp-fold)" if recipe.splice else ")"))
                    continue
                for v in variants:
                    pos_prob, y_true = res[v]
                    run_points[v].append(metrics.point_metrics(pos_prob, y_true))

            for v in variants:
                if run_points[v]:
                    per_task[v][size][name] = {
                        **{k: float(np.mean([p[k] for p in run_points[v]])) for k in
                           (*metrics.METRIC_KEYS, "frac_pos_query")},
                        "n_runs": len(run_points[v]),
                        "runs": run_points[v],
                        "ec": ec_by_task.get(name, "unknown"),
                    }
            if any(run_points[v] for v in variants):
                bits = []
                for v in variants:
                    if run_points[v]:
                        t = per_task[v][size][name]
                        bits.append(f"{v}: dAUPRC={t['delta_auprc']:+.4f} AUROC={t['roc_auc']:.4f}")
                print(f"    support {size:3d}: " + "  |  ".join(bits))

    # ---- aggregate + report -------------------------------------------------------------------
    summaries = {v: metrics.summarise(per_task[v], support_sizes) for v in variants}
    for v in variants:
        metrics.print_report(v, summaries[v], support_sizes, use_ema, recipe.atoms_desc)
    metrics.print_ab(summaries, variants, support_sizes)
    if repair_status_counts or task_errors:
        print(f"\nSMILES repair_status across all evaluated tasks: {repair_status_counts}")
    if task_errors:
        print(f"[warn] {len(task_errors)} task(s) SKIPPED ENTIRELY ({recipe.atoms} features unbuildable): {list(task_errors)}")

    # ---- save --------------------------------------------------------------------------
    stem = Path(script_file).stem
    name_out = f"{stem}.shard{args.shard_index}of{args.num_shards}.json" if sharded else f"{stem}.json"
    out = args.output_json or (output_dir / name_out)

    per_task_json = {
        v: {str(size): {tn: {k: val[k] for k in (*metrics.METRIC_KEYS, "frac_pos_query", "n_runs", "ec")}
                        for tn, val in per_task[v][size].items()}
            for size in support_sizes}
        for v in variants
    }
    payload = {
        "dataset": "FS-Mol",
        "atom_features": ATOM_FEATURES_TAG[recipe.atoms],
    }
    if recipe.atoms == "x9_plus_extra":
        payload.update({
            "atom_feature_columns": list(env.atom_features.X_PLUS_EXTRA_FEATURE_COLUMNS),
            "n_extra_features": n_cols,
            "smiles_repair": "smiles_repair.robust_mol (regex phosphorus-valence fix + "
                             "permissive-sanitize fallback) -- no samples dropped",
            "repair_status_counts": repair_status_counts,
            "task_errors": task_errors,
        })
    if recipe.splice:
        payload.update({
            "per_mol_splice": per_mol_mode,
            "fp_len_raw": data.FP_LEN,
            "fp_fold": fp_fold,
            "permol_feature_width": permol_width,
            "splice": "ef.encode_molebert_as_groups -> concat onto `pooled` after output_proj "
                      "(ef.pooling_gnn_forward_with_molebert / forward_pass_with_molebert)",
        })
    payload.update({
        "edges": ("bidirectional" if bidirectional else "as_stored") + "+isolated_self_loop",
        "protocol": "fs_mol.utils.test_utils.eval_model (StratifiedTaskSampler, test_size_or_ratio=None)",
        "task_list_json": str(task_list_json),
        "n_test_tasks": len(test_task_names),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "metric": "delta_auprc = average_precision_score(y_query, P_active) - query_positive_rate",
        "checkpoint_path": str(pooler_checkpoint),
        "checkpoint_step": checkpoint.get("step"),
        "use_ema": use_ema,
        "backbone_checkpoint": str(env.bt.CHECKPOINT_PATH),
        "support_sizes": support_sizes,
        "num_runs": args.num_runs,
        "seed": args.seed,
        "max_query": args.max_query,
    })
    if not recipe.flat_json:
        payload["variants"] = variants
    # per-shard task-selection filters only -- whether THIS shard covers its full round-robin slice is a
    # merge-time concern (merge.py's own `missing` check), not folded in here.
    payload["paper_comparable"] = args.limit_tasks is None and args.max_query is None and args.tasks is None
    if recipe.flat_json:
        (only,) = variants
        payload["summary"], payload["per_task"] = summaries[only], per_task_json[only]
    else:
        payload["summary"], payload["per_task"] = summaries, per_task_json
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved to {out}")
