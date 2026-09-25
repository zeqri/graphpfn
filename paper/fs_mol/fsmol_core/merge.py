"""Combine the per-shard result JSONs of a sharded evaluation run (evaluate.py, --num-shards N) into
one paper-comparable result: the same schema a single unsharded run would have produced over the full
157-task test set.

Each shard JSON covers a disjoint round-robin slice of the test tasks (tasks[shard_index::num_shards])
and carries its own `per_task` block. The merge
  1. loads every shard matching --glob (or the explicit paths given positionally),
  2. checks the config fields that change what the numbers MEAN agree across shards (variants, support
     sizes, num_runs, seed, per-molecule feature / fold, atom features, edges, task list) and refuses to
     merge shards from different configs,
  3. unions `per_task` (per variant, if the run had variants) and errors if a task appears twice
     (overlapping shards would double count),
  4. recomputes `summary` (the Figure-2a curve + Table-2 EC breakdown) over the union,
  5. sums `repair_status_counts` and unions `task_errors` (x9 recipe) so the merged JSON still answers
     "were any samples dropped" for the full run.

CAVEAT: the per-shard JSONs only persist each task's PER-RUN-AVERAGED metrics, to keep them small. For an
EC category that ends up with just ONE task after merging, the unsharded run reports that task's error
as the std across its own runs; here it is the across-task aggregate with n=1 (std = sem = 0) -- noted in
the output JSON's "note" field. `all_enzymes` and every EC category with >1 task are unaffected.
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
from pathlib import Path

from . import metrics

# Config fields that change what the numbers MEAN -- must agree exactly across shards (a field absent
# from every shard trivially agrees).
STRICT_FIELDS = (
    "variants", "support_sizes", "num_runs", "seed", "per_mol_splice", "fp_fold", "permol_feature_width",
    "atom_features", "atom_feature_columns", "n_extra_features", "edges", "metric", "task_list_json",
)
# Fields that should agree but are only warned about (e.g. absolute paths that may differ by node).
SOFT_FIELDS = ("checkpoint_path", "backbone_checkpoint", "use_ema")
# Per-shard bookkeeping that does not carry over to the merged file.
PER_SHARD_FIELDS = ("n_test_tasks", "shard_index", "num_shards", "paper_comparable", "summary", "per_task",
                    "checkpoint_step", "repair_status_counts", "task_errors")


def _fmt(v):
    return json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v


def parse_args(eval_stem: str, output_dir: Path, description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("shard_json", nargs="*", type=Path, help="Explicit shard JSON paths (alternative to --glob).")
    p.add_argument("--glob", type=str, default=None, help="Glob pattern (quoted) matching shard JSON files.")
    p.add_argument("--output-json", type=Path, default=None, help=f"Default: {output_dir}/{eval_stem}_merged.json")
    p.add_argument("--allow-partial", action="store_true",
                   help="Don't fail if fewer distinct shard_index values than num_shards are found "
                        "(e.g. merging while some array tasks are still running / were skipped).")
    return p.parse_args()


def main(eval_stem: str, output_dir: Path, atoms_desc: str, description: str) -> None:
    args = parse_args(eval_stem, output_dir, description)

    paths = list(args.shard_json)
    if args.glob:
        paths += [Path(p) for p in sorted(globmod.glob(args.glob))]
    if not paths:
        raise SystemExit("no shard files given -- pass paths positionally or via --glob 'pattern'")
    paths = sorted(set(paths))
    print(f"merging {len(paths)} shard file(s):")
    for p in paths:
        print(f"  {p}")

    shards = []
    for p in paths:
        if not p.exists():
            raise SystemExit(f"shard file not found: {p}")
        shards.append(json.loads(p.read_text()))

    # ---- config consistency -------------------------------------------------------------------
    ref = shards[0]
    for field in STRICT_FIELDS:
        vals = {_fmt(s.get(field)) for s in shards}
        if len(vals) > 1:
            raise SystemExit(f"shards disagree on '{field}': {vals} -- refusing to merge mismatched runs")
    for field in SOFT_FIELDS:
        vals = {_fmt(s.get(field)) for s in shards}
        if len(vals) > 1:
            print(f"  [warn] shards disagree on '{field}': {vals} (proceeding anyway)")

    has_variants = "variants" in ref            # runs without a splice have a flat per_task[size] schema
    variants = ref["variants"] if has_variants else [None]
    support_sizes = ref["support_sizes"]
    num_shards_seen = sorted({s.get("num_shards") for s in shards})
    shard_indices_seen = sorted({s.get("shard_index") for s in shards})
    if len(num_shards_seen) > 1:
        raise SystemExit(f"shards report different --num-shards values: {num_shards_seen}")
    declared_num_shards = num_shards_seen[0] if num_shards_seen else len(shards)
    missing = sorted(set(range(declared_num_shards)) - set(shard_indices_seen))
    if missing and not args.allow_partial:
        raise SystemExit(
            f"missing shard_index {missing} out of {declared_num_shards} -- pass --allow-partial to "
            f"merge anyway (result will under-cover the test set)"
        )
    if missing:
        print(f"  [warn] --allow-partial: missing shard_index {missing} -- merged result is PARTIAL")

    # ---- union per_task (per variant), checking for duplicate/overlapping task coverage -------
    per_task: dict = {v: {s: {} for s in support_sizes} for v in variants}
    task_to_shard: dict[str, Path] = {}
    for path, shard in zip(paths, shards):
        for v in variants:
            for size in support_sizes:
                block = shard.get("per_task", {})
                block = (block.get(v, {}) if has_variants else block).get(str(size), {})
                for task_name, task_metrics in block.items():
                    key = f"{v}@{task_name}@{size}"
                    if key in task_to_shard:
                        raise SystemExit(
                            f"task '{task_name}' @ variant '{v}' @ support {size} appears in both "
                            f"{task_to_shard[key]} and {path} -- overlapping shards, refusing to merge "
                            f"(would double count)"
                        )
                    task_to_shard[key] = path
                    per_task[v][size][task_name] = task_metrics

    n_test_tasks = len({k.split("@", 1)[1].rsplit("@", 1)[0] for k in task_to_shard}) if task_to_shard else 0
    print(f"\ncovered {n_test_tasks} distinct test tasks across {len(shards)} shard(s) "
          f"(declared num_shards={declared_num_shards}, shard_index seen={shard_indices_seen})")
    for v in variants:
        for size in support_sizes:
            print(f"  " + (f"[{v}] " if has_variants else "") + f"support {size:3d}: {len(per_task[v][size])} tasks")

    # ---- repair_status_counts (summed) + task_errors (unioned) ---------------------------------
    repair_status_counts: dict[str, int] = {}
    task_errors: dict[str, str] = {}
    for shard in shards:
        for k, n in (shard.get("repair_status_counts") or {}).items():
            repair_status_counts[k] = repair_status_counts.get(k, 0) + n
        task_errors.update(shard.get("task_errors") or {})
    if task_errors:
        print(f"  [warn] {len(task_errors)} task(s) were SKIPPED ENTIRELY across shards "
              f"(atom features unbuildable): {list(task_errors)}")

    # ---- recompute summary (across-task aggregation only -- see module docstring CAVEAT) ------
    summaries = {v: metrics.summarise(per_task[v], support_sizes) for v in variants}
    for v in variants:
        metrics.print_report(v or "all", summaries[v], support_sizes, ref.get("use_ema", True), atoms_desc)
    if has_variants:
        metrics.print_ab(summaries, variants, support_sizes)

    all_paper_comparable = all(s.get("paper_comparable") is not False for s in shards) and not missing
    payload = {k: val for k, val in ref.items() if k not in PER_SHARD_FIELDS}
    if "repair_status_counts" in ref or "task_errors" in ref:
        payload.update({"repair_status_counts": repair_status_counts, "task_errors": task_errors})
    payload.update({
        "n_test_tasks": n_test_tasks,
        "num_shards": declared_num_shards,
        "shards_merged": [str(p) for p in paths],
        "shard_indices_covered": shard_indices_seen,
        "paper_comparable": all_paper_comparable,
        "note": "by_ec entries with n_tasks==1 use across-task SEM (std=0, sem=0) rather than the "
                "unsharded script's within-task std-across-runs, because shard files do not persist "
                "per-run values. all_enzymes and every EC category with >1 task are unaffected.",
    })
    if has_variants:
        payload["summary"] = summaries
        payload["per_task"] = {v: {str(size): per_task[v][size] for size in support_sizes} for v in variants}
    else:
        (only,) = variants
        payload["summary"] = summaries[only]
        payload["per_task"] = {str(size): per_task[only][size] for size in support_sizes}

    out = args.output_json or (output_dir / f"{eval_stem}_merged.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved merged result to {out}")
