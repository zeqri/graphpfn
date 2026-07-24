# Graph-level (whole-molecule) regression via learnable pooling

This document records the design conversation and the implementation that came out of
it: adding graph-level ("predict one value per molecule") task support to the
GraphPFN/LimiX pipeline, on top of a pipeline that was previously strictly node-level.

## 1. Starting point: how does LimiX map raw features into the model?

Before any of this, the first questions were about the existing backbone: where raw
per-atom/per-node features get mapped into `embed_dim`. Traced through:

- `FeaturesTransformer.forward` (`vendor/limix/model/transformer.py`) reshapes raw
  features into `(batch, seq, n_feature_groups, features_per_group)` groups, then runs
  `x_preprocess` (NaN handling, outlier removal, normalization) followed by `encoder_x`.
- `encoder_x` is built by `get_x_encoder` (`vendor/limix/model/encoders.py`), wrapping
  `MaskEmbEncoder`: each scalar feature goes through a small per-feature MLP (or a
  learned mask embedding if it's NaN), then a `fusion_network` collapses each
  atom/node's group of features down to one `embed_dim`-sized vector.

This is the mapping every later architectural decision built on top of.

## 2. Reviewing the existing pooling prototype

`dev/pool_icl_real_limix.py` (pre-existing, one-off sanity check) already proved a
learnable-pooling architecture works: per-atom `encoder_x` embedding → distance-biased
geometric attention among one molecule's own atoms (`GeometricAtomAttention`, reusing
`EdgeDistanceEncoder` from `lib/graphpfn/model.py`) → learnable attention-pool to one
per-molecule token (`AttentionPooler`) → the real backbone's graph-adapter repurposed as
full attention among pooled molecule tokens → the real `y_decoder`.

Review found it worked, but with two real weaknesses:
- **Unvectorized**: loops over molecules one at a time in Python.
- **Per-molecule normalization**: `x_preprocess` is called separately per molecule (as
  few as ~2-30 atoms), a small-sample-statistics problem — the same class of bug the
  script's own docstring documents for label standardization (`standardize_by_context`,
  fixed by using population-wide stats instead of context-only).

## 3. Is the pooling mechanism related to the geometric graph adapter in `model.py`?

Traced the relationship between `GeometricAtomAttention`/`AttentionPooler` (dev script)
and `GraphPFNGraphAttentionModule` (the real graph adapter used inside the 12 wrapped
LimiX layers):

- Both use the **same** `EdgeDistanceEncoder` (RBF expansion + zero-init linear →
  per-head additive attention bias) — literally shared code.
- But they differ architecturally: `GraphPFNGraphAttentionModule` is zero-init by
  default (a strict residual no-op until trained), pre-norm, repeated at all 12 layers,
  supports a sparse `dgl.DGLGraph` backend; `GeometricAtomAttention` is not zero-init,
  post-norm, runs once, dense-only.
- `AttentionPooler` (plain Set-Transformer-style pooling) has no geometric component at
  all.
- The post-pooling reuse of the real 12-layer adapter runs it in *non-geometric* mode
  (`edge_distance=None`) — full attention among molecule tokens, no distance signal,
  since there's no real inter-molecule distance to encode.

## 4. Does the pooled representation depend on both node features and edge distances?

Yes: `encoder_x` embeds node features first (no attention yet), then
`GeometricAtomAttention` mixes in distance-biased neighbor information, then pooling
aggregates all of a molecule's (already geometry-informed) atoms into one token. Caveat:
`EdgeDistanceEncoder`'s linear layer is zero-initialized, so **at initialization** only
the topology (which atoms are bonded) shapes the representation — the actual distance
*magnitude* has zero effect until that encoder is trained away from zero.

## 5. Would this work if transferred into the main pipeline?

Surveyed `bin/graphpfn/pretrain.py`, `lib/graphpfn/model.py`, `lib/graphpfn/prior/`:

- The main pipeline is **strictly node-level** — `TaskType` only has
  REGRESSION/BINCLASS/MULTICLASS, `evaluate_dataset` asserts `is_transductive`, and
  nothing outside the dev script even references pooling or virtual nodes. There was no
  existing graph-level mechanism to "compete with" or plug into.
- `GraphPFN.forward`/`FeaturesTransformer.forward` are single-graph-per-call
  (`x.shape[0] == 1`), but that's not a "one molecule" limit — the `seq` dimension can
  (and does, via `sample_multi_graph`'s `dgl.batch`) hold many molecules' atoms in one
  call, block-diagonal, with zero cross-molecule edges.
- Two architecturally different ways to add graph-level support were identified: a
  **virtual super-node** (star-connect one extra node per molecule, reuse the node-level
  machinery unchanged) vs. **learnable pooling** (the dev script's approach, more
  explicit/novel, more new code). The user chose **learnable pooling**, deliberately,
  after the tradeoffs were laid out.

## 6. Planning: architecture and open questions

Entered plan mode. Researched (via parallel Explore/Plan agents):
- `pretrain.py`'s full training loop (step_fn, gradient accumulation, checkpointing,
  DDP wrapping, TaskType/is_transductive gating).
- Prior-sampling internals (`multi_graph.py`, `gnn_scm.py`, `postprocessing/features.py`)
  — specifically whether pooling could be vectorized instead of looped.
- The existing QM9 finetune scripts' pattern (streaming pools, checkpoint format,
  context/query split at molecule granularity).

Key design decisions made during planning, several revised after user pushback:

- **DDP data sampling**: each rank samples its own dataset independently (own RNG
  stream), not a master-samples-then-scatter protocol — the existing
  `GraphPriorSamplerDDP`'s padded fixed-shape scatter design doesn't fit variable-size
  `dgl.batch` graphs anyway.
- **Pooled inter-molecule attention**: stays strictly full/complete-graph for v1 (no
  real inter-molecule graph exists to justify anything sparser).
- **`n_graphs` scale** (initially set to the reference config's 10-30): the user
  pointed out this is far too few "in-context samples" compared to the main pipeline's
  1000-5000 node range, and could degenerate context to a single molecule given
  `train_ratio` as low as 0.05. Scaled up to **1000-2000** to fix both problems (later
  reduced again after a real-world OOM — see §11).
- **Pooling must be sparse, not dense**: at `n_graphs=1000-2000`, a dense
  `(n_graphs, total_atoms)` masked attention (the naive vectorization of the dev
  script's loop) would cost ~2.9 billion entries per step, ~99.9% of which get masked
  out anyway. Redesigned as a sparse bipartite `dgl` graph (edges only from each atom to
  its own molecule's pooling-query node), reusing the same `dgl.ops.u_dot_v` /
  `edge_softmax` / `u_mul_e_sum` pattern the real geometric adapter already uses —
  O(total_atoms) instead of O(n_graphs × total_atoms).
- **Cross-molecule attention boundaries**: user asked to confirm attention in "graph
  pooling" doesn't cross molecule boundaries. Confirmed: atom-level geometric attention
  and the pooling step itself are strictly per-molecule (real bonds never cross
  molecules; pooling edges are atom→its-own-molecule's-query-node-only). The *only*
  place attention crosses molecule boundaries is the post-pooling stage (the repurposed
  graph adapter running full attention among already-pooled tokens) — intentional, since
  that's the actual in-context-learning mechanism.
- **Label generation** (virtual-node trick, promoted from the dev script): user asked
  whether the graph-level label can be made to depend *only* on the atom features, not
  on the virtual node's own randomness. Added a `zero_cause_mask` parameter to
  `sample_attributes_gnn` so the virtual node's own exogenous "cause" input is zeroed
  before the SCM runs — its label becomes a function purely of the real atoms,
  propagated through the same graph-convolution machinery real per-atom labels use.
  Confirmed (from the reference config's `graph_conv_ratio=1.0` pinning) that every
  node's hidden state, at every layer, is *already* pure neighbor-aggregation with no
  self-fallback — reinforcing that the label is a genuine, structured function of the
  graph, learnable by a model whose own architecture mirrors the same
  message-pass-then-readout shape.

## 7. The approved plan

Final plan (in full at `~/.claude/plans/ticklish-foraging-clock.md`): vectorized
pooling model + new synthetic-label helper + new dedicated pretraining entrypoint,
training on the synthetic multi-graph molecule prior with periodic real-QM9 evaluation
interleaved (zero-shot/ICL, no gradient from QM9). Deferred: a full QM9 finetuning
script (gradient updates from real QM9 labels).

## 8. Implementation

Files created/modified, in dependency order:

1. `lib/graphpfn/prior/attributes/gnn_scm.py` — added `zero_cause_mask` parameter to
   `sample_attributes_gnn` (opt-in, default `None`, every other caller unaffected).
2. `lib/graphpfn/prior/attributes/graph_level.py` (new) —
   `sample_graph_level_labels_via_virtual_node`: vectorized virtual-node star-connection
   + SCM label generation, no Python loop over molecules.
3. `lib/graphpfn/prior/graph_level.py` (new) — `sample_graph_level_dataset`: composes
   `sample_multi_graph` + the label helper + `process_features` (called once over the
   whole batch, fixing the per-molecule small-sample-normalization problem on the
   label-generation side too).
4. `lib/graphpfn/pooling.py` (new) — `GeometricAttentionStack` (reuses
   `GraphPFNGraphAttentionModule` unmodified, on the raw batched atom graph),
   `SparseAttentionPooler` (sparse bipartite pooling, see §6), `GraphLevelGraphPFN`
   (wraps a `GraphPFN`, does the vectorized context-first reorder, and drives the real
   `FeaturesTransformer` machinery — `x_preprocess`, `encoder_x`, `add_embeddings`,
   `mixed_y_embedding`, `transformer_encoder`, `y_decoder` — unchanged).
5. `lib/graphpfn/qm9_data.py` (new) — `load_qm9`, `QM9_TARGETS`,
   `sample_one_qm9_dataset` (promoted from the dev script, real bond distances from
   `data.pos`), producing the same batched-tensor shape as the synthetic sampler.
6. `bin/graphpfn/pretrain_graph_level.py` (new) — training entrypoint, mirroring
   `pretrain.py`'s DDP/EMA/checkpointing/gradient-accumulation scaffolding, with its own
   `step_fn`/`eval_fn` and periodic real-QM9 evaluation.
7. `exp/graphpfn/pretrain/multigraph_molecule_graph_level_pooling/pretrain.toml` (new)
   — config, reusing the reference experiment's prior subtree unchanged except
   `n_graphs`.
8. `exp/graphpfn/pretrain/multigraph_molecule_graph_level_pooling/submit.sh` (new) —
   SLURM submission script (4 GPUs, `torch.distributed.run`).

### Bugs found and fixed while implementing (not anticipated by the plan)

- **`shuffle_nodes` destroys `dgl.batch`'s `batch_num_nodes()` metadata.** The generic
  `sample_graph` dispatcher's multi-graph branch calls `shuffle_nodes`, which rebuilds
  the graph via a bare `dgl.graph(...)` — silently losing per-molecule membership info.
  Fixed by calling `sample_multi_graph` directly, bypassing the dispatcher (safe here:
  atom order never matters for this permutation-invariant pipeline).
- **Hardcoded `float32` dtype for `y_full`** in `GraphLevelGraphPFN.forward` broke under
  a full-precision (`float64`) correctness test. Fixed to follow `atom_features.dtype`.
- **Context/query split retry loop retried the split, not the whole dataset.** For some
  `(n_graphs, train_ratio)` draws every possible split is invalid (e.g.
  `round(6*0.05)=0` train molecules, deterministically, regardless of the random
  permutation) — retrying `context_query_split` alone on the same sampled dataset could
  loop forever. Fixed by consolidating into one retry that resamples the whole dataset.
- **`edge_head` has no Python default** (`GraphPFN.__init__` requires it explicitly),
  but TOML can't express `None`. Fixed by defaulting to `None` in
  `pretrain_graph_level.main` unless the config opts in.
- **`bin/go.py` always calls the pretrain function with a `profiler=` keyword
  argument** — `pretrain_graph_level.main` didn't accept one and would have crashed
  immediately when launched via the standard `bin/go.py <toml>` cluster entrypoint.
  Fixed by adding the parameter (threaded through to `profiler.step()` per
  gradient-accumulation micro-step, matching `pretrain.py`'s own convention).

## 9. Verification

- `SparseAttentionPooler` is bit-exact against a manual dense-attention reference
  (after fixing a k/v-splitting convention bug in the *test itself*, not the pooler:
  the real code reshapes into per-head chunks before splitting k/v, not the other way
  around).
- **Zero cross-molecule leakage** at the atom-encoding/geometric-attention/pooling
  stages, confirmed by hooking `encoder_x`'s output directly: perturbing one molecule's
  atoms by a large amount changed *only* that molecule's own atoms' embeddings, exactly,
  everywhere else bit-identical.
- **In-context learning is functioning**: perturbing a context molecule's revealed label
  measurably changes query predictions.
- **Permutation invariance** to molecule ordering, down to float32 machine precision
  (1.1e-6), after isolating a real confound: LimiX's own `add_embeddings` samples a
  fresh random positional embedding on *every* forward call (`subortho` type) — even two
  calls on the identical unpermuted input differ by the same amount as the "permuted"
  comparison did, until both calls are seeded identically.
- **Gradients flow** to both new submodules (`geometric_attention`, `pooler`).
- **Full pipeline end-to-end**: `pretrain_graph_level.main()` ran for several epochs on
  CPU with the real LimiX-16M checkpoint (scaled-down `n_graphs` for speed) — gradient
  accumulation, EMA, checkpoint save/resume, and periodic evaluation on both synthetic
  data and real QM9 molecules all worked, producing finite losses and sensible
  (untrained-baseline) R² scores. Also verified via the actual `bin/go.py <toml>`
  invocation path (not just a bespoke test harness).
- **Not tested** (no hardware available during development): actual multi-GPU DDP
  execution, and performance/memory at the full production `n_graphs=1000-2000` scale.
  This gap mattered — see §11.

## 10. Answering practical questions about the run

- **Evaluation size**: runs once per epoch. 8 synthetic datasets (same prior/scale as
  training — context ~50-1000 molecules originally, see §11 for the revised range) + 8
  QM9 datasets (`n_graphs` 10-30, `train_ratio` fixed at 0.3 — context ~3-9 molecules).
  Both counts are config-controlled (`n_synthetic_eval_datasets`, `qm9_eval.*`).
- **Logging**: `training_log.jsonl` (one JSON line appended per epoch: step, synthetic
  score, qm9 score, lr, loss), `report.json` (full report, overwritten each epoch),
  `checkpoint.pt` (torch binary, not JSON, for resuming).
- **QM9 target**: `cv` (heat capacity at 298.15K), configurable via `qm9_eval.target`
  from the standard 19 QM9 targets.

## 11. First real cluster run: OOM, diagnosis, and fix

First real submission (4×40GB GPUs) got through the pre-training baseline eval
successfully, then crashed on the very first real training step:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.66 GiB.
GPU 1 has a total capacity of 39.49 GiB ... 38.64 GiB memory in use.
```

Traceback pinpointed `SparseAttentionPooler.forward` → `dgl.ops.u_mul_e_sum`. Root
cause: at `n_graphs=1000-2000`, `total_atoms` could reach ~120,000, and `n_features` in
this prior goes up to 100 (→ up to 50 feature groups). `SparseAttentionPooler` pads
every atom row with a dummy zero query and every query row with dummy zero
keys/values to keep one uniformly-shaped node-indexed tensor for `dgl.ops` — roughly
doubling memory on top of an already-large `(total_atoms, n_groups, n_heads, d_head)`
tensor, on top of the 12-layer backbone and geometric-attention stage sharing the same
GPU.

Two paths were possible: rework `SparseAttentionPooler` to avoid the padding waste
(e.g. a `dgl` heterograph with separate atom/query node types), or shrink `n_graphs` as
an immediate mitigation. **Chose the quick mitigation**: `n_graphs` reduced from
`1000-2000` to **`100-300`** — total atoms now ~1,500-18,000 (a 20-80x reduction),
comfortably under what OOM'd, while still meaningfully larger context (~5-150
molecules) than the original reference config's 10-30. Re-verified the smoke test still
passes after the change. The proper pooler memory fix remains a documented follow-up
(see the comment in `pretrain.toml` above the `n_graphs` setting) if `n_graphs` needs to
go back up later.

Also noted: resubmitting with `--continue` after changing the config would immediately
fail `create_report`'s config-mismatch check against the stale `report.json` from the
crashed run — the old output directory needs clearing (or `--force` instead of
`--continue`) before resubmitting.

## 12. Second real run: works, but slow — profiling the actual bottleneck

With `n_graphs=100-300`, a resubmitted 4-GPU run got well past the OOM point: 6+ epochs
completed, loss dropped 0.94→0.60, R² rose from strongly negative to positive on both
synthetic (~0.38-0.46) and QM9 (~0.17-0.28) — real learning signal, ICL working as
intended. But it was slow: exact epoch-boundary timestamps pulled from the slurm log
showed a **remarkably consistent ~55 minutes/epoch** (100 accumulated steps × 20
micro-steps = 2000 forward/backward passes + one eval). At `n_steps=10000`,
`epoch_size=100`, that's ~92 hours (~3.8 days, ~16 resubmissions of the 6-hour job) to
reach the configured target — a genuine throughput problem, not impatience.

Rather than guess at a fix, profiled the actual cost breakdown (CPU-only, no GPU
available during development, but informative for relative proportions and for ruling
things out):

- **Dataset sampling is only ~9%** of per-step time on CPU (mean 0.655s sampling vs.
  6.53s forward+backward) — contradicts an initial hypothesis that missing
  prefetching (this pipeline has none; the node-level pipeline's `GraphPriorSampler`
  has dedicated multiprocess workers, this one samples synchronously) was the dominant
  cost. It's real, but a minority.
- **A forward-pass breakdown** (single trial, n_graphs=228, 9189 atoms) found
  `dgl.graph()` construction is negligible (0.0004s — ruling out another hypothesis),
  while cost splits roughly: `transformer_encoder` (12 backbone layers) 44%,
  `encoder_x` 25%, `geometric_attention` (sparse `dgl.ops`) 16%, `pooler` (sparse
  `dgl.ops`) 10%.
- The nuance: `transformer_encoder`/`encoder_x` are dense matrix ops that should
  accelerate a lot on real GPU hardware, while the sparse `dgl.ops` stages
  (`geometric_attention`+`pooler`, 26% combined) and the sampling cost (9%) are less
  likely to shrink proportionally — meaning once the dense compute shrinks on GPU, the
  sparse ops and sampling both likely become a *larger* fraction of GPU wall-clock time
  than they were of CPU wall-clock time. No single silver-bullet bottleneck.

## 13. Adding background prefetching (partial fix) + a GPU profiling script (pending)

Decided on both: implement prefetching now (a clear, if partial, win — recovers the
~9%+ of wall-clock time that's currently pure idle-GPU wait), and instrument the next
real cluster run with `torch.profiler` to settle the sparse-`dgl.ops`-on-GPU question
empirically instead of extrapolating further from CPU numbers.

**`GraphLevelPriorSampler`** (new, `lib/graphpfn/prior/graph_level.py`) mirrors the
existing `GraphPriorSampler`'s `DataLoader`-based worker mechanism (same
`n_workers`/`prefetch_factor` convention) exactly, with one necessary adaptation:
worker subprocesses return a plain-tensor `GraphLevelSample` dict, not a
`dgl.DGLGraph` — `dgl` graphs aren't reliably picklable across the multiprocessing IPC
boundary (the original node-level sampler sidesteps this the same way, representing
graphs as plain edge tensors and reconstructing `dgl.DGLGraph` back in the main
process). `raw_sample_to_graph` reconstructs it locally. Wired into
`pretrain_graph_level.py`'s `step_fn` (the hot path); `eval_fn` keeps sampling
synchronously since it's rare (once/epoch). New `[base_config.sampler]` toml section
(`n_workers`, `prefetch_factor`) — set conservatively (6 workers/rank × 4 ranks = 24
background processes, leaving headroom in the 48-CPU allocation shared across ranks).

**Two more real bugs found and fixed while building this:**

- **The exact same `batch_num_nodes()`-destroying bug from §8, reintroduced.**
  `raw_sample_to_graph`'s graph reconstruction (`dgl.graph((src, dst), num_nodes=...)`)
  is the identical anti-pattern `shuffle_nodes` has — it silently collapsed every
  prefetched sample to `n_graphs=1`, caught by an isolated debug script showing
  `n_graphs=1` on every single draw regardless of the configured range. Fixed with
  `graph.set_batch_num_nodes(counts)` (the `GraphLevelSample`'s `counts` field, carried
  across the worker boundary specifically for this).
- **Spawn-multiprocessing requires an `if __name__ == "__main__":` guard** in whatever
  script launches the training run — without it, a `spawn`-context worker re-imports
  and re-executes the launching script's top-level code from scratch. `bin/go.py`
  already has this guard (safe in production); an ad-hoc test script used during
  verification didn't, and needed the same fix.

Verified end-to-end again after both fixes (CPU, scaled-down `n_graphs`, via both a
bespoke harness and the real `bin/go.py <toml>` path) — training runs cleanly with
prefetching enabled.

**`dev/profile_graph_level_pooling.py`** (new) — a GPU-only diagnostic script (needs a
real GPU allocation to run; not runnable during development) that (1) profiles a few
real training steps with `torch.profiler` and prints the top ops by CUDA time, to
settle whether the sparse `dgl.ops` stages or the dense backbone dominate real GPU
time, and (2) times several steps with `n_workers=0` vs. the configured prefetching, on
the same GPU, to directly quantify prefetching's actual benefit (the CPU estimate was
necessarily indirect). **Not yet run** — needs the user to execute it on an actual GPU
node/allocation and share the output.

## Current status

- Pipeline is implemented, unit- and integration-verified on CPU/single-process, and
  runs cleanly through the real `bin/go.py` cluster entrypoint.
- A real 4-GPU run confirmed the pipeline trains correctly (loss decreasing, R² rising
  on both synthetic data and real QM9) once past the initial OOM.
- `n_graphs` is currently `100-300` (reduced from the originally-intended `1000-2000`
  after the OOM) — a known, documented compromise, not the final word on scale.
- Background dataset prefetching (`GraphLevelPriorSampler`) is now wired into the
  training hot path — expected to recover some, not necessarily all, of the observed
  ~55-min/epoch throughput, per the profiling in §12.
- **Not yet verified**: real GPU-side timing (the profiling script in §13 hasn't been
  run yet — this is the next concrete step to actually know whether prefetching alone
  fixed the slowness or whether the sparse-`dgl.ops`/`SparseAttentionPooler`-padding
  rework flagged in §11 is also needed); whether a full multi-GPU DDP run can now
  complete within the 6-hour walltime; the deferred full QM9 finetuning script.

## Where things live

- Model/pooling code: `lib/graphpfn/pooling.py`, `lib/graphpfn/qm9_data.py`
- Prior/label generation + prefetching sampler: `lib/graphpfn/prior/graph_level.py`,
  `lib/graphpfn/prior/attributes/graph_level.py`, `lib/graphpfn/prior/attributes/gnn_scm.py`
- Training entrypoint: `bin/graphpfn/pretrain_graph_level.py`
- Config + cluster script: `exp/graphpfn/pretrain/multigraph_molecule_graph_level_pooling/`
- GPU profiling script (pending a run): `dev/profile_graph_level_pooling.py`
- Original prototype this was built from: `dev/pool_icl_real_limix.py`
- Full original implementation plan: `~/.claude/plans/ticklish-foraging-clock.md`
