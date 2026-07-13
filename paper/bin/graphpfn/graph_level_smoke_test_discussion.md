# GraphPFN Graph-Level Regression via Virtual Node — Discussion

File under discussion: `paper/bin/graphpfn/graph_level_smoke_test.py`

## Purpose of the smoke test

Checks whether GraphPFN's existing virtual-node + adapter mechanism (no new
model code) can compute and regress a whole-graph structural statistic (mean
node degree), using one **fixed deterministic** labeling function shared
across all sampled graphs. This deliberately removes the in-context-learning
requirement — the model learns the function in its adapter weights across
training steps, rather than inferring it per-dataset from labeled context.
The goal is to isolate the *architecture* question ("can structure reach a
query node and get decoded into a scalar") from the *prior-learning*
question.

## Code walkthrough

1. **`parse_args`** — CLI knobs for graph size range, SBM group count,
   feature dim, and training-stability knobs: `n_steps`,
   `n_grad_accum_steps` (default 16), `n_warmup_steps` (default 100).
2. **`sample_graph_config`** — draws random `n_nodes`, `avg_degree`,
   `n_groups`; returns a `GraphConfig` for an SBM (stochastic block model)
   sampler. Each call produces a structurally different graph.
3. **`average_degree`** — the one fixed label function: mean in-degree of
   the graph. Deterministic and simple by design.
4. **`calibrate_label_stats`** — samples 500 graphs up front to get
   mean/std of `average_degree` over the graph-generation config, so the
   label can be z-scored (`predict-prior-mean baseline MSE=1.0` becomes a
   meaningful reference).
5. **`build_virtual_node_input`** — appends one new node to the graph,
   wires it bidirectionally to every real node, gives it a zero feature
   vector, and sets `train_mask=False` only for it (all real nodes are
   context). Real nodes get `y_train` set to i.i.d. noise (not zeros) —
   a constant label would itself be a copyable in-context pattern, letting
   the model cheat instead of routing real structural information to the
   virtual node.
6. **`main()` training loop** — each optimizer step samples 16 fresh graphs,
   runs each through the model, reads `out["predictions"][virtual_idx]` as
   the model's prediction, computes MSE against the standardized label, and
   backprops `loss / n_grad_accum_steps` per graph before one
   `optimizer.step()`. LR follows a linear warmup over the first 100 steps.

Important nuance: node features are **not** derived from the prior — only
graph *structure* comes from `sample_graph`. Features are pure
`torch.randn` noise, uncorrelated with structure or label. The virtual node
never receives the label as input; the label is only used *outside* the
model to compute the loss against the model's prediction at the virtual
node's output position.

## Observed training run

```
Calibrating label statistics (mean degree) over the graph prior...
  label mean=4.4441, std=1.3122
n_trainable_params=3,564,288
step    20 | loss=0.9982 | corr=+0.036
step   100 | loss=0.4710 | corr=+0.746
step   200 | loss=0.0385 | corr=+0.981
step   420 | loss=0.0260 | corr=+0.987
```

**Assessment:** healthy convergence. Loss starts at the predict-the-mean
baseline (1.0), drops sharply once LR warmup completes (~step 100), and
correlation climbs to ~0.98–0.99 by step 300–420. The residual floor
(~0.02–0.05, with some step-to-step bounce) is expected: it comes from (a)
calibration noise in `label_mean`/`label_std` (only 500 samples), and (b)
genuine SGD variance from averaging over just 16 graphs per step with no LR
decay.

### Why grad-accumulation + LR warmup stabilized a previously fluctuating loss

Per the docstring: *"raw single-sample SGD is noisy enough to collapse the
model into predicting the label's marginal mean regardless of input."*

- **Without grad accumulation** (1 graph/step): each graph has different
  `n_nodes`/`avg_degree`/`n_groups`, so single-sample gradients are
  high-variance and contradictory step to step. Under enough noise, the
  optimizer's safest low-loss strategy is to just output the marginal mean.
  Averaging over 16 graphs before the update cuts gradient variance by
  ~4× (∝ 1/√16).
- **Without LR warmup**: jumping straight to `lr=1e-3` on step 1, with a
  randomly-initialized adapter and high-variance gradients, produces large
  destructive early steps that can knock the adapter into the "predict the
  mean" collapse before it finds a sensible region. The linear ramp from
  ~0 to `1e-3` over the first 100 steps avoids this.

Both are copied from `pretrain.toml`'s config, i.e. reusing the same fix
that the real pretraining loop already needed for the same reason.

## Is this a PFN?

**No, not as currently trained — deliberately.** A PFN (Prior-Fitted
Network) is defined by in-context learning: at inference time it infers an
unseen task's function from labeled context, with no weight updates, having
been meta-trained across many varying tasks from a prior. This smoke test
breaks that on purpose:

- `average_degree` is the *same fixed function* for every graph — no
  varying task to infer.
- Context nodes' `y_train` is i.i.d. noise, not real labeled examples of
  the target function — no informative context to learn from even if the
  model wanted to.
- The only way to reduce loss is to bake `structure → mean degree` into the
  adapter weights across training steps (ordinary amortized regression of a
  frozen feature extractor), not per-instance in-context inference.
- `freeze_tfm=True`: the transformer backbone (presumably pretrained
  elsewhere as a genuine node-level PFN) is frozen; only the ~3.5M-param
  adapter is trained here, on one fixed function.

The frozen backbone may originally have been trained as a real PFN, but
this run doesn't exercise or validate that ICL property at the graph level.
To actually test "is this a graph-level PFN," the label function would need
to vary per task and be inferable from real labeled context graphs (not
noise) — the natural next step beyond this smoke test.

## Can graph-level regression be done without a virtual node?

Investigated the actual model architecture (`paper/lib/graphpfn/model.py`,
imported by the smoke test) to answer this.

### Architecture is a hybrid of two mixing pathways

1. **ICL sequence attention** (`GraphPFNLayerWrapper` → LimiX base layer,
   `model.py:325-351`, `vendor/limix/model/layer.py:551-583`): full
   attention, but **query → context only**
   (`x_kv = x[:, :eval_pos]`, `model.py` / layer.py). Query nodes never
   attend to each other via this path; context attends only to context.
   Crucially, this attention is **not edge-restricted** — a query attends
   to *all* context nodes regardless of graph adjacency.
2. **Graph-conv** (`GraphPFNGraphAttentionModule`, `model.py:354-418`):
   edge-restricted, DGL-masked 1-hop message passing per layer
   (`model.py:186`, `391-394`). `train_mask` has **no effect** on this
   adjacency mask — it's built purely from graph edges regardless of
   train/query status, so query nodes *can* message-pass with each other,
   but only through real edges, one hop per layer (up to ~12 layers total).

### No existing pooling/readout mechanism

Searched `src/graphpfn` and `paper/lib/graphpfn` for "pool", "readout",
"graph_level", "global_mean", "virtual node" — the virtual-node
construction in the smoke test itself is the *only* graph-level
aggregation mechanism in the codebase. Heads (`feat_head`, `edge_head`) are
strictly per-node/per-edge; `out["predictions"]` is always shape
`[n_nodes]` (one row per node, train rows zeroed as placeholders, unpermuted
back to original order via `inv_perm`, `model.py:200-212`).

### Yes, a no-virtual-node alternative is architecturally possible...

Since `train_mask` already just splits real nodes into context/query
independent of graph structure, you could mark a genuine subset of real
nodes as query (`train_mask=False`) and the rest as context, then mean-pool
`out["predictions"]` at the query indices post-hoc as the graph-level
scalar — no model code changes needed. Constraint: can't make *all* real
nodes query (ICL cross-attention needs a non-empty context set).

### ...but it has a real information-flow cost

Any real node marked **query** is thereby *excluded* from context, so:

- Its structural signal becomes invisible to *every other query node's*
  ICL attention (pathway 1) — a query node can no longer see it as a
  key/value at all.
- The only remaining path for that node's info to reach other query nodes
  is graph-conv (pathway 2) — edge-restricted, hop-limited (~12 layers).
  If query nodes are more than ~12 hops apart, or the query fraction is
  large, that information effectively doesn't propagate.

The **virtual-node** design sidesteps this entirely: keeping *all* real
nodes as context guarantees the query (virtual node) sees every node's
info in a single hop via full ICL attention, independent of graph size or
diameter.

**Practical trade-off:** a no-virtual-node variant would need a very small
query set relative to context (e.g. leave-one-out: one held-out real node
per forward pass) to preserve near-complete context coverage — but that
means one query prediction per forward pass instead of many, requiring
multiple forward passes per graph to build a graph-level mean-pooled
estimate (vs. the virtual node's single pass). This is a genuine
accuracy/cost trade-off, and arguably the strongest argument for keeping
the virtual-node design rather than dropping it.

**Open question for next step:** prototype the no-virtual-node
(leave-one-out) variant to empirically measure the degradation, or continue
building on the virtual-node approach — e.g. extending to a real
in-context-learning setup (varying label function per task, real labeled
context graphs instead of noise) to actually test the graph-level PFN
property.
