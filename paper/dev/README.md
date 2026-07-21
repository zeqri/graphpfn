# Multi-Graph Priors for GraphPFN Pretraining

This documents the exploratory work in this session: changing GraphPFN's
pretraining prior from sampling **one large graph per synthetic dataset** to
sampling **several smaller graphs combined into one dataset**, and pushing
that idea toward resembling real organic-molecule datasets.

Everything here is additive — the original single-graph prior and its
pretraining config (`exp/graphpfn/pretrain/main/`) are untouched and behave
exactly as before.

## 1. Baseline: the original single-graph prior

As background, we first sampled and visualized what the paper's original
prior actually produces: one graph per dataset (1000-5000 nodes), generated
by a multi-level stochastic block model with preferential attachment
(multi-level-sbm-with-pa) — designed to resemble social/information
networks, with dense community "blocks" and a preferential-attachment
periphery.

`sample_prior_graph.py` samples and plots this directly:

![Single-graph prior samples](graph_prior_samples.png)

## 2. Multi-graph prior: combining several graphs into one dataset

The core idea: instead of one big graph per training step, sample `n_graphs`
independent smaller graphs and combine them via disjoint union (so the
adjacency is block-diagonal — no edges between them). Since the graph
adapters' message passing / attention only ever look at 1-hop neighbors,
this required no changes to the model, attribute generation, or training
loop at all — only to how the graph *structure* is sampled.

The total node budget was kept proportional to the original single-graph
setup (scaled down by the mean number of graphs), so per-step compute stays
comparable.

`sample_multi_graph_prior.py` visualizes the raw structure (sub-graphs
colored by which graph they belong to):

![Multi-graph prior samples](multi_graph_prior_samples.png)

`visualize_prior_sample.py` goes further and samples a *full* dataset
(features, labels, train/test split) through the exact same pipeline used
during training, for any config by name — showing what the model actually
sees, not just the structure:

![Full dataset sample from the multigraph prior](prior_sample_multigraph.png)

This config is trained with `run.sh`, and logs loss/synthetic-score to a
`training_log.jsonl` file per training step.

**On anticipated effects:** before running this, we discussed the expected
impact of combining more, smaller graphs per dataset instead of one large
graph — namely, that it dilutes both the graph structure's local
neighborhood richness and the number of labeled ("train") nodes available
per sub-graph, which should make the task harder and likely show up as a
higher loss/lower synthetic-score plateau than the single-graph baseline,
even though the *training mechanics* are unaffected.

## 3. Resembling organic-molecule datasets

A natural real-world analogue for "many graphs per dataset" is a molecular
dataset: many separate molecules (graphs), each with relatively few atoms,
where molecules *within one dataset* tend to be similar in size to each
other (unlike the first multi-graph prior above, where each sub-graph's size
was drawn totally independently).

To capture that, sub-graph sizing was changed to two levels of randomness:
- `base_n_nodes`: one shared "typical size" for the whole dataset, resolved
  once per dataset (not per sub-graph) — think of it as that dataset's
  characteristic molecule size.
- each sub-graph's actual size is then `base_n_nodes` jittered by only a
  small relative amount, so sub-graphs in one dataset stay close in size to
  each other.

Combined with many more (10-30), smaller (15-60 node), sparser sub-graphs
per dataset, `visualize_molecule_prior.py` shows both the graph layout and a
size-consistency panel (each sub-graph's size against the shared target ±
jitter band):

![Molecule-like prior samples, multi-level-sbm-with-pa](molecule_prior_samples_multi-level-sbm-with-pa.png)

This variant is trained with `run_molecule.sh`.

## 4. Does the structural generator itself matter?

The multi-level-sbm-with-pa generator above is purpose-built to resemble
*social networks* — dense community blocks plus a preferential-attachment
periphery, only *probabilistically* connected. That's a poor structural
match for real molecules, which are connected by construction, close to
trees, and degree-bounded by valence rather than having power-law "hub"
atoms.

To check this empirically, `visualize_molecule_prior.py` supports swapping
the sub-graph generator while holding everything else (graph count, target
size, jitter, density) fixed, for a controlled, apples-to-apples comparison:

**Plain SBM** (no hierarchy, no preferential attachment):

![Molecule-like prior samples, plain SBM](molecule_prior_samples_sbm.png)

**Random geometric graph** (nodes as points in latent space, edges between
nearby points — a different generative family entirely):

![Molecule-like prior samples, geometric](molecule_prior_samples_geometric.png)

Both alternatives turned out to match the "consistent size within a
dataset" goal noticeably worse than multi-level-sbm-with-pa, since sparser,
smaller graphs from either generator disconnect more easily and lose more
nodes to trimming.

## 5. A purpose-built molecule-like generator: tree + ring closure

Since none of the existing generators were designed with molecular topology
in mind, we implemented a new one: a degree-capped random tree (connected
*by construction* — no trimming needed at all) plus a small number of
ring-closing edges between nodes that are already close to each other in the
tree, so ring sizes stay realistic. This directly encodes the two defining
structural properties of organic molecules: guaranteed connectivity and
valence-bounded degree.

![Molecule-like prior samples, tree-with-rings](molecule_prior_samples_tree-with-rings.png)

This produced by far the tightest size consistency of any generator tried —
every sample's sub-graph sizes landed at or near the target band, with no
outliers. This variant is trained with `run_molecule_tree.sh`.

## 6. Adding geometry: valence, bond order, and QM9 bond distances

Section 5's tree-with-rings generator gives valid, molecule-like *topology*,
but topology alone can't distinguish molecules that share a local shape: CH2
and CO2 are both a degree-2 star, but their bond lengths differ. The next
step, prototyped standalone in `sample_geometric_molecule_prior.py` (does
not touch the training pipeline — pure sampling + visualization, like the
other `dev/*.py` scripts), was to layer atom identity and 3D bond distance on
top, targeting distance first and deliberately deferring bond angles/3D
placement.

This went through three iterations, each caught by actually sampling and
looking at the output rather than reasoning about it in the abstract:

1. **Atom types sampled independently of topology.** Draw an element per
   node (weighted toward QM9's composition: H > C > O > N > F) *after* the
   degree-capped tree was already built, then sample each edge's distance
   from a per-element-pair Gaussian fit to QM9 bond-length statistics.
   Distances matched the QM9 targets closely (validated by aggregating
   thousands of sampled edges against the target `N(mu, sigma)`), but the
   topology was chemically inconsistent — since degree was capped by one
   global `max_degree=4` regardless of which atom ended up on a node, a
   degree-4 nitrogen (valence 3) with 3 hydrogens already attached could
   still pick up a 4th bond.

2. **Per-node valence caps + bond order.** The real fix: sample the element
   *first*, derive a valence budget from it (H/F=1, O=2, N=3, C=4), and grow
   the tree/rings so each new edge consumes a sampled *bond order* (1/2/3,
   weighted toward single bonds) worth of valence from both endpoints — not
   just 1. This is what makes valence, not degree, the capped quantity: a
   C#C triple bond consumes 3 of carbon's 4 valence units through a single
   edge (degree contribution 1), leaving exactly 1 unit for one more bond —
   giving that carbon degree 2 overall, matching acetylene H-C#C-H. Bonds
   touching H/F are automatically forced to order 1, with no special-casing,
   since remaining valence for those atoms never exceeds 1.

   This mostly worked, but elements were still drawn independently and
   competed uniformly at random for attachment slots during tree growth.
   Since H alone is ~52% of atoms and saturates instantly (valence 1), the
   pool of nodes with spare capacity emptied out fast as the tree grew,
   forcing a "degenerate" fallback (attach anyway, exceeding valence) on
   about 10% of attachments — visible directly in a diagnostic panel
   aggregating realized valence usage per element against each element's
   cap across hundreds of sampled sub-graphs.

3. **Skeleton first, hydrogens after.** Remove the competition entirely:
   sample only the heavy atoms (C/N/O/F) and grow the valence-capped
   tree/rings *among those*, then walk the finished skeleton and attach one
   explicit hydrogen leaf per unit of *leftover* valence on each heavy atom.
   H can now only ever be added, never compete for a slot, so valence
   violations for H become structurally impossible, and the heavy-only
   valence distribution (mean ~3.3) is far less skewed than the mixed one
   (mean ~1.8), making the growth-phase fallback rare too. This dropped
   degenerate attachments from ~10% to ~0.05% of all nodes, confirmed by
   re-running the same valence-usage diagnostic.

   One more calibration was needed here: the sub-graph template's
   `avg_degree` (3.0-6.0) had been tuned for the old semantics where most
   nodes were degree-1 hydrogens dragging the average down. Applied directly
   to the heavy-only skeleton, it asked for 3-6 heavy-heavy bonds per heavy
   atom — far denser than real small organic molecules — which consumed
   nearly all valence in heavy-heavy ring bonds and starved hydrogen
   generation (skeletons came out as dense fused meshes with almost no H).
   Giving the heavy skeleton its own, lower avg-degree range (1.8-2.6,
   closer to real heavy-atom-only connectivity) fixed this.

Bond *distance* is still sampled per edge from the per-element-pair QM9
prior below (mean, std in Angstrom), with a generic fallback
(`N(1.45, 0.15)`) for element pairs QM9 doesn't cover here (e.g. H-H, F-F,
F-N, F-O, O-O) — and does not yet depend on the sampled bond order (a C=C
and a C-C currently draw from the same C-C prior). That's the natural next
refinement, deferred so this step could focus on getting the valence
bookkeeping right first.

| Bond | Mean (Å) | Std (Å) |
|---|---:|---:|
| C-C | 1.5042 | 0.0719 |
| C-F | 1.3332 | 0.0115 |
| C-H | 1.0929 | 0.0068 |
| C-N | 1.3918 | 0.1011 |
| C-O | 1.3689 | 0.0906 |
| H-N | 1.0128 | 0.0059 |
| H-O | 0.9645 | 0.0030 |
| N-N | 1.3311 | 0.0382 |
| N-O | 1.3982 | 0.0503 |

Final result — node color = element, edge color = sampled distance
(blue=short, red=long), edge width = sampled bond order:

![Geometric molecule prior gallery](geometric_molecule_prior_gallery.png)

Distance-validation panel — sampled distances aggregated per element pair
across 300 sub-graphs, overlaid on the target QM9 `N(mu, sigma)`:

![Geometric molecule prior distance validation](geometric_molecule_prior_validation.png)

Valence-validation panel — realized degree and realized valence usage per
element, aggregated across 300 sub-graphs, checked against each element's
valence cap (should never exceed the red dashed line):

![Geometric molecule prior valence validation](geometric_molecule_prior_valence.png)

**Status (as of section 6):** validated in isolation only — this was still a
`dev/` sandbox script, not wired into the training pipeline. Sections 7-11
below cover the next session, which did wire it in.

## 7. Diagnosing why `molecule_skeleton` (valence-bounded) plateaus early

Running `multigraph_molecule_valence_bounded` (the trained version of
section 5/6's valence-capped skeleton, `lib/graphpfn/prior/graphs/
molecule_skeleton.py`) side by side with the earlier `multigraph`
(multi-level-sbm-with-pa) config showed two distinct problems, not one:

1. **Slower early convergence.** In the first ~900 steps, `synthetic`
   accuracy rose 0.42→0.50 for `multigraph_molecule_valence_bounded` vs.
   0.42→0.57 for `multigraph`.
2. **A hard plateau, not just slow convergence.** Over the *full* run (7600+
   steps), `multigraph_molecule_valence_bounded` climbed to only ~0.52 and
   flatlined by step ~2000, while `multigraph` kept slowly climbing to ~0.58
   by step 5600 — a real, ~6-point gap in ceiling, reached much earlier.

The diagnosis, grounded in the actual code rather than just the config diff:

- **Low degree ⇒ the SCM's GNN aggregation degenerates into a copy, not an
  average.** `attributes/layers.py`'s `MixedGraphLinear` mixes each hidden
  unit between a linear transform and a graph convolution
  (`GCNConv`/`SAGEConv`/`GTConv`). For a degree-1 node (H leaves, ~52% of all
  atoms), `SAGEConv`'s mean/min/max of one value is just that value, and
  `GCNConv`'s degree-normalization reduces to an unweighted copy — none of
  these average over anything. Since `attributes/common.py`'s
  `extract_features_and_labels` draws both the observed features **and** the
  label itself from the same pool of SCM hidden units (any of which may have
  gone through this graph-mixing), this directly weakens how much the
  *label* can depend on real aggregated structure, not just the features.
- **Low *and narrow* degree ⇒ nothing left to discover after the easy part.**
  `avg_degree` for the heavy-atom skeleton was `log_uniform(1.8, 2.6)` — a
  narrow band, versus `multigraph`'s `mixed_log_uniform(1.5, 500)`. Attention
  over exactly one neighbor is a forced, degenerate case with no weighting
  decision to learn; once the adapter learns "copy your one neighbor if
  you're a leaf, lightly mix up to 4 neighbors if you're a heavy atom",
  every subsequent sampled graph looks like a minor variation on the same
  theme, with no harder regime to keep learning from.
- **Compounded by no block/community structure.** `multi_level_sbm_with_pa.py`
  builds correlated multi-node neighborhoods via `n_groups`/`offdiagonal_coef`
  (nonzero clustering coefficient — median 0.16 per the paper's Table 4); a
  tree-plus-occasional-ring skeleton has clustering coefficient ≈ 0 by
  construction, so there's no "next level" of pattern above the immediate
  1-hop neighborhood for the adapter to move on to.

**Caveat:** this was inferred from the code/config, not from directly
logging realized degree histograms per batch — that would be the way to
confirm it empirically (see section 11).

## 8. Does the plateau matter for downstream transfer? QM9NMR results

Before changing anything, we checked whether a lower/earlier-plateauing
*synthetic* ICL metric during pretraining actually predicts worse downstream
transfer — it's a proxy metric, and the paper itself only uses it for
post-hoc monitoring, never model selection.

`bin/graphpfn/qm9nmr_carbon_only_streaming_full_finetune.py` (full DDP
fine-tune, backbone + adapters unfrozen, on real QM9NMR carbon-only isotropic
shielding, molecule-disjoint train/test split) gave a first real data point:

| step | R² | MAE (ppm) |
|---|---:|---:|
| 0 (zero-shot) | 0.665 | 20.55 |
| 300 | 0.925 | 9.86 |
| 2000 | 0.963 | 7.32 |
| 3900 | 0.971 | 6.42 |

Two takeaways:
- **Zero-shot transfer from a purely synthetic, chemistry-agnostic
  pretraining prior is already non-trivial** (R²=0.665 on real NMR data).
- **Finetuning kept improving smoothly past step 3900 with no sign of the
  pretraining-time plateau reappearing** — consistent with the pretraining
  plateau being a property of the *synthetic prior's* diversity ceiling, not
  a hard cap on what the model can eventually represent, since full
  finetuning doesn't need to generalize across the whole synthetic
  distribution, only fit this one real task directly.
- **Open caveat:** GraphPFN's architecture is node-level only (graph-level
  regression/classification is an explicit limitation in the paper's own
  Appendix A) — this only works because QM9NMR carbon shielding is a
  genuine per-atom (node-level) target. We also don't yet have a controlled
  ablation of this same finetuning recipe against the `multigraph`
  (SBM-based) checkpoint or a randomly-initialized-adapter baseline, so we
  can't yet say the molecule-specific prior transfers *better* than a
  mismatched one — only that it transfers non-trivially.

## 9. Adding a geometric (bond-distance) prior to the real pipeline

This directly completes section 6's TODO list. Scope decisions made before
implementing: **distance only** for this pass (bond order deferred), a
**fresh pretraining run from LimiX** (not warm-started from the existing
`multigraph_molecule_valence_bounded` checkpoint, for a cleaner ablation),
and **RBF expansion + a small zero-init linear layer** (SchNet/EGNN-style
continuous filter) for how distance enters attention, rather than a raw
linear term.

**Prior generation** (`lib/graphpfn/prior/graphs/molecule_skeleton.py`):
ported the `BOND_PRIOR` per-element-pair QM9 Gaussian fits from
`dev/sample_geometric_molecule_prior.py`; `sample_molecule_skeleton` now
samples one distance per edge and exposes it as `graph.edata["distance"]`,
gated behind a new `compute_distances` flag (default `True`; see section 11
for why the flag exists).

**A real bug found along the way:** `sample_graph`'s dispatcher
(`lib/graphpfn/prior/graphs/__init__.py`) re-runs `to_simple`/
`extract_largest_component`/`shuffle_nodes` after every sampler — and all
three rebuild the `dgl.DGLGraph` via `dgl.graph((src, dst), ...)` without
copying `edata`, exactly the gap section 6 flagged. Since
`molecule_skeleton`'s output is already guaranteed simple/connected by
construction, we special-cased it to skip the redundant rebuild (mirroring
the existing `multi-graph` special case) and made `shuffle_nodes` itself
edata-safe generically (safe because permutation never reorders/drops
edges).

**Threading `edge_distance` through the rest of the pipeline**, following the
existing SSL/MGM edge pathway's convention (a plain tensor parallel to
`edges`, not trying to keep a `dgl.DGLGraph` object alive through the whole
pipeline):
- `PriorDataset`/`PriorDatasetBatch` (`prior_typings.py`): new
  `edge_distance` field (zero-filled placeholder for non-geometric
  priors — provably a no-op via softmax shift-invariance, see below).
- `graph_then_attributes.py`/`attributes_then_graph.py`: extract
  `edge_distance` from `graph.edata` right where `edges` itself is
  extracted, before the graph object is discarded.
- `sampler.py`: `_pad_and_batch` pads it like `edges`;
  `GraphPriorSamplerDDP` broadcasts/scatters it as a 7th field alongside
  the existing 6.
- `bin/graphpfn/pretrain.py`'s `step_fn`: reattaches it as
  `graph.edata["distance"]` *before* the SSL edge-masking step, so
  `graph.remove_edges` keeps it in sync automatically.

**SCM (synthetic label generation):** new `GeometricConv`
(`attributes/layers.py`) — an RBF-expansion continuous-filter aggregation,
weights randomly initialized like every other SCM layer (never trained) —
added as a new `"geometric-rbf"` `conv_type` choice, so pretraining labels
can genuinely depend on bond geometry, not just topology.

**Model (real graph adapter, `model.py`):** new `EdgeDistanceEncoder` (RBF
expansion + zero-init linear) adds a per-head additive distance bias to
`GraphPFNGraphAttentionModule`'s attention scores, on *both* the sparse-DGL
path (`dgl.ops.u_dot_v`/`edge_softmax`) and the dense `SDPAInput` path
(`F.scaled_dot_product_attention`, extended with a dense `edge_distance`
tensor). Zero-init guarantees this is a strict no-op until trained.

**New experiment:** `exp/graphpfn/pretrain/multigraph_molecule_geometric/`
(+ `run_molecule_geometric.sh`) — same structure as
`multigraph_molecule_valence_bounded` but with `"geometric-rbf"` added to
`scm.conv_type`'s choices and `compute_distances = true` set explicitly.

**Downstream:** `qm9nmr_carbon_only_streaming_full_finetune.py` updated to
compute real bond distances from QM9's `mol.pos` (3D coordinates) and thread
them through each induced training subgraph via DGL's `EID` edge-id
mapping, so a geometric-prior checkpoint sees real distances during
finetuning too, not just pretraining.

**Verification performed (actually executed, not just read):**
- Prior side: `sample_molecule_skeleton` produces finite, positive,
  shape-aligned distances; the full `sample_graph` dispatcher preserves
  `edata` through `multi_graph`/`shuffle_nodes` (confirming the bug fix);
  non-geometric samplers (`tree-with-rings`) are unaffected;
  `graph_then_attributes.sample_dataset` runs end-to-end with
  `"geometric-rbf"`.
- Model side: `EdgeDistanceEncoder` is exactly zero at init, but changes
  the module's output once its weights are perturbed (confirms it's wired
  up, not dead code) — verified on both the sparse and dense attention
  paths; `_pad_and_batch` round-trips `edge_distance` correctly aligned
  with `edges` columns.
- Full integration: a real `GraphPFN.forward` pass (random-init backbone,
  no network download needed) succeeds and produces finite predictions
  both with and without `edata["distance"]` present.
- Downstream: `build_carbon_only_pool`/`extract_micro_dataset` produce
  real, positive, QM9-range bond distances (0.96-1.79 Å) that match the
  synthetic `BOND_PRIOR`'s range, and cross-checking via the `EID` mapping
  confirms subgraph distances exactly match the full pool's.

## 10. Early training: geometric vs. valence-bounded

Comparing `multigraph_molecule_geometric`'s `synthetic` accuracy against
`multigraph_molecule_valence_bounded`'s at the same steps:

| step | `valence_bounded` | `geometric` |
|---|---:|---:|
| 200 | 0.419 | 0.409 |
| 900 | 0.501 | 0.458 |
| 1500 | 0.512 | 0.467 |

At step 200 this gap is noise (still deep in the 1000-step LR warmup, and
the zero-init property mathematically guarantees the new distance pathway
can't yet be the cause). By step 1500 it's a real, growing gap that the
zero-init argument no longer fully explains, since real gradient steps have
accumulated. Two plausible, non-alarming explanations discussed:
- **Not a controlled comparison** — `molecule_skeleton.py` now draws an
  extra random distance per edge, shifting the entire subsequent RNG
  stream even under the same nominal `seed=0`, so this is two different
  random realizations of a similar prior, not the same datasets plus one
  added feature.
- **Added curriculum diversity** — widening the SCM's `conv_type` set from
  5 to 6 options alone (independent of geometry specifically) makes the
  aggregate prior a more diverse mixture of generative mechanisms, which is
  intrinsically harder to fit uniformly early on.

Not yet implemented: logging the `distance_encoder`'s weight-norm directly,
which would disambiguate "geometric pathway is actively being learned and
costing gradient budget" from "it's barely moving, and the gap is just
added task diversity."

## 11. A simple, non-geometric ablation: widening `avg_degree`

Directly testing section 7's "low-and-narrow degree" diagnosis, without
geometry: a new experiment,
`exp/graphpfn/pretrain/multigraph_molecule_valence_bounded_wider_degree/`
(+ `run_molecule_valence_bounded_wider_degree.sh`), changes exactly **one**
value from `multigraph_molecule_valence_bounded` — the molecule-skeleton
sub-graph's `avg_degree` range, `log_uniform(1.8, 2.6)` →
`log_uniform(1.8, 4.0)` — leaving ring size, `heavy_atom_fraction`,
`conv_type`, and everything else untouched, to isolate this one variable
cleanly.

**A second real bug, found while verifying this ablation:** since section
9's `molecule_skeleton.py` change computes distances *unconditionally*, and
`model.py`'s attention bias activates whenever `graph.edata["distance"]` is
present (regardless of `conv_type`), this "no geometry" ablation would have
silently exercised the geometric attention pathway anyway. Fixed by making
`compute_distances` an explicit, per-experiment flag: `false` for
`wider_degree` (and also applied to `multigraph_molecule_valence_bounded`'s
own config, to protect against a future `--continue` restart silently
picking up geometry — this doesn't affect that job while it's already
running, since its config is already parsed into memory), `true` for
`multigraph_molecule_geometric`.

**Verified:** both non-geometric configs now correctly produce all-zero
`edge_distance`. `wider_degree` produces measurably denser graphs — mean
directed-edges/node 2.35 vs. 2.08 (~1.13x) — modest because
`heavy_atom_fraction` (0.42) is unchanged, so ~58% of nodes are still H
leaves permanently stuck at degree 1, diluting the effect on overall
average degree even though the heavy-atom-only subgraph got meaningfully
denser (median avg-degree roughly 2.16→2.68).

**Status:** created and verified, not yet submitted as a training job.

## 12. Instrumentation + a conv_type-frequency ablation for the geometric run

While waiting on the runs above, two more additions:

**Diagnostic logging** (`bin/graphpfn/pretrain.py`): a `distance_encoder_norm`
field now gets written to every `training_log.jsonl` entry — the L2 norm of
every `EdgeDistanceEncoder` parameter across all layers. Since that encoder
is zero-init and only ever receives a forward/backward pass when
`graph.edata["distance"]` is present, this norm is a direct readout of
whether the geometric pathway is actually being learned, rather than
inferring it indirectly from the aggregate `synthetic` curve. It's logged
unconditionally (present in the model regardless of config), which doubles
as a passive regression check: it should stay *exactly* 0.0 for
`valence_bounded`/`wider_degree` for their entire runs (no forward pass ever
reaches it there), and only move for `geometric`-family configs. This only
takes effect for newly-started processes, not already-running jobs.

**New experiment:** `exp/graphpfn/pretrain/multigraph_molecule_geometric_upweighted/`
(+ `run_molecule_geometric_upweighted.sh`) — a one-variable ablation of
`multigraph_molecule_geometric`: adds a `conv_type.weights` override
(`geometric-rbf = 5.0`, others default to 1.0) to `config.py`'s `_sample_choice`
weighting mechanism, raising `"geometric-rbf"`'s sampling probability from
uniform 1-in-6 (~16.7%) to ~50%. Motivated directly by section 10's
observation that `geometric`'s early climb lagged `valence_bounded`'s more
than the zero-init argument alone could explain — one candidate cause being
that distance-dependent labels were simply too rare to give
`EdgeDistanceEncoder` a strong gradient signal. Verified by sampling 400
configs: `geometric-rbf` now lands at 45.5%, and end-to-end dataset sampling
(real, positive `edge_distance`) still works.

**Status:** created and verified, not yet submitted as a training job.

## 13. Bond-order-dependent distances

Prompted by a direct question during review: does `BOND_PRIOR`'s pooling of
all bond orders into one Gaussian per element pair actually hide real
variation, or do the pooled means already look similar enough that it
doesn't matter? Checked empirically against real QM9 data (all 130,831
molecules, distances grouped by RDKit's actual bond type, not textbook
estimates) rather than assumed:

| Pair | Pooled (old `BOND_PRIOR`) | Single | Double | Triple |
|---|---|---|---|---|
| C-C | mean 1.5039, std **0.0724** | 1.5214 ± 0.0385 | 1.3624 ± 0.0309 | 1.2039 ± 0.0063 |
| C-N | mean 1.3918, std **0.1016** | 1.4254 ± 0.0721 | 1.3117 ± 0.0305 | 1.1563 ± 0.0016 |
| C-O | mean 1.3688, std **0.0905** | 1.4135 ± 0.0334 | 1.2055 ± 0.0090 | — |
| N-N | mean 1.3315, std **0.0376** | 1.3479 ± 0.0224 | 1.2897 ± 0.0359 | — |
| N-O | mean 1.3997, std **0.0513** | 1.4069 ± 0.0382 | 1.2280 ± 0.0084 | — |

The pooled *means* do look similar across pairs (1.33-1.50 Å) — but that's
an artifact of pooling, not evidence of little real variation: each pair
actually spans a wide range once split by order (e.g. C-O: 1.41 Å single vs.
1.21 Å double), and the pooled mean just lands wherever that pair's specific
single/double mixing ratio in QM9 happens to average out to. The std is the
real tell, and it collapses once split (e.g. C-O's double-bond std drops
10x, from 0.0905 pooled to 0.0090). Bonds to H or F (valence 1, so
`_sample_bond_order` forces order=1 always) were already "pure" single-order
fits and don't need splitting — confirmed by their already-tiny pooled std
(0.003-0.012 Å) — only C-C/C-N/C-O/N-N/N-O (both endpoints valence ≥2) show
this mixing.

Wired in as a new **opt-in, default-off** `bond_order_aware_distances` flag
on `sample_molecule_skeleton` (`lib/graphpfn/prior/graphs/
molecule_skeleton.py`), specifically so this couldn't affect any
already-running or already-defined experiment:
- `BOND_PRIOR_BY_ORDER: {(element_pair, bond_order): (mean, std)}` — the
  measured values above, covering only pairs where both elements have
  valence ≥2 (H/F-involving pairs fall back to the existing pooled
  `BOND_PRIOR`, which is already correct for them).
- `_sample_bond_distances` takes an optional `bond_order_by_pair` dict; when
  `None` (the default), it takes the *exact same code path* as before this
  change — no new randomness is consumed, so RNG-stream-sensitive
  comparisons (see section 10) aren't disturbed for any config that doesn't
  opt in.
- `sample_molecule_skeleton` builds `bond_order_by_pair` only when the new
  flag is explicitly `True` — heavy-heavy edges from the already-computed
  `order_by_pair`, heavy-H edges implicitly order 1 (exact, not
  approximate, since H's valence of 1 makes any other order impossible).

**New experiment:** `exp/graphpfn/pretrain/multigraph_molecule_geometric_bond_order_aware/`
(+ `run_molecule_geometric_bond_order_aware.sh`) — a one-variable ablation of
`multigraph_molecule_geometric`, adding only `bond_order_aware_distances = true`.

**Verified:**
- With the new flag omitted or explicitly `False`, two identically-seeded
  calls to `sample_molecule_skeleton` produce byte-identical distances —
  confirming zero behavioral change for every pre-existing config.
- All 4 pre-existing configs (`valence_bounded`, `geometric`,
  `geometric_upweighted`, `wider_degree`) still sample end-to-end without
  error, with `edge_distance` unchanged (all-zero for the non-geometric
  two, same real range as before for the geometric two).
- The new config's aggregate distance variance is measurably lower than the
  pooled version's (std 0.1963 vs. 0.1982 over 21k+ sampled edges) — a
  modest-looking aggregate reduction because most edges in any molecule are
  H-bonds (already order-pure, identical either way); the real effect is
  concentrated in the C-C/C-N/C-O/N-N/N-O subset, where it's large (see
  the per-pair table above).

**Status:** created and verified, not yet submitted as a training job.

## Summary of experiment configs

| Config | Sub-graphs / dataset | Sub-graph size | Structural generator | Geometry | Launch script |
|---|---|---|---|---|---|
| `main` | 1 | 1000-5000 | multi-level-sbm-with-pa | no | (original, unmodified) |
| `multigraph` | 2-4 | 84-1667, independent per sub-graph | multi-level-sbm-with-pa | no | `run.sh` |
| `multigraph_molecule` | 10-30 | 15-60, shared target ± small jitter | multi-level-sbm-with-pa | no | `run_molecule.sh` |
| `multigraph_molecule_tree` | 10-30 | 15-60, shared target ± small jitter | tree-with-rings | no | `run_molecule_tree.sh` |
| `multigraph_molecule_valence_bounded` | 10-30 | 15-60, shared target ± small jitter | molecule-skeleton, avg_degree 1.8-2.6 | no (`compute_distances=false`) | `run_molecule_valence_bounded.sh` |
| `multigraph_molecule_geometric` | 10-30 | 15-60, shared target ± small jitter | molecule-skeleton, avg_degree 1.8-2.6 | **yes** (`compute_distances=true`, `geometric-rbf` conv_type @ 1/6 weight, pooled distances) | `run_molecule_geometric.sh` |
| `multigraph_molecule_geometric_upweighted` | 10-30 | 15-60, shared target ± small jitter | molecule-skeleton, avg_degree 1.8-2.6 | **yes** (same, but `geometric-rbf` conv_type @ ~50% weight) | `run_molecule_geometric_upweighted.sh` |
| `multigraph_molecule_geometric_bond_order_aware` | 10-30 | 15-60, shared target ± small jitter | molecule-skeleton, avg_degree 1.8-2.6 | **yes** (same as `geometric`, but `bond_order_aware_distances=true`) | `run_molecule_geometric_bond_order_aware.sh` |
| `multigraph_molecule_valence_bounded_wider_degree` | 10-30 | 15-60, shared target ± small jitter | molecule-skeleton, avg_degree **1.8-4.0** | no (`compute_distances=false`) | `run_molecule_valence_bounded_wider_degree.sh` |

## Where things live

- **Prior sampling code:** `lib/graphpfn/prior/graphs/` — `multi_graph.py`
  (combining sub-graphs into one dataset), `tree_with_rings.py`, and
  `molecule_skeleton.py` (valence-capped skeleton, now with real QM9-derived
  bond distances gated by `compute_distances`) are the new building blocks;
  everything else in that directory is the paper's original code.
- **SCM / attribute generation:** `lib/graphpfn/prior/attributes/layers.py`
  — `GeometricConv` (new, RBF continuous-filter, `"geometric-rbf"`
  `conv_type`) alongside the original `GCNConv`/`SAGEConv`/`GTConv`.
- **Model / real graph adapter:** `lib/graphpfn/model.py` —
  `EdgeDistanceEncoder` (new) inside `GraphPFNGraphAttentionModule`; the
  same class is used regardless of whether a given graph carries distance
  data (falls back to a no-op when it doesn't).
- **Training configs:** `exp/graphpfn/pretrain/<name>/pretrain.toml`
- **Visualization scripts:** `dev/*.py` (this directory) — each is
  self-contained and safe to run anytime (CPU-only, no GPU/model involved),
  independent of any training job that happens to be running.
- **Geometry prototype:** `dev/sample_geometric_molecule_prior.py` (section
  6) — superseded by the real, wired-in version in `molecule_skeleton.py`
  (section 9), but left as-is as a standalone reference/visualization tool.
- **Downstream evaluation:** `bin/graphpfn/qm9nmr_carbon_only_streaming_full_finetune.py`
  — full DDP finetune on real QM9NMR carbon-only shielding, now
  geometry-aware (real `mol.pos`-derived distances).
