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

## Summary of experiment configs

| Config | Sub-graphs / dataset | Sub-graph size | Structural generator | Launch script |
|---|---|---|---|---|
| `main` | 1 | 1000-5000 | multi-level-sbm-with-pa | (original, unmodified) |
| `multigraph` | 2-4 | 84-1667, independent per sub-graph | multi-level-sbm-with-pa | `run.sh` |
| `multigraph_molecule` | 10-30 | 15-60, shared target ± small jitter | multi-level-sbm-with-pa | `run_molecule.sh` |
| `multigraph_molecule_tree` | 10-30 | 15-60, shared target ± small jitter | tree-with-rings | `run_molecule_tree.sh` |

## Where things live

- **Prior sampling code:** `lib/graphpfn/prior/graphs/` — `multi_graph.py`
  (combining sub-graphs into one dataset) and `tree_with_rings.py` (the new
  molecule-like generator) are the two new building blocks; everything else
  in that directory is the paper's original code, unmodified.
- **Training configs:** `exp/graphpfn/pretrain/<name>/pretrain.toml`
- **Visualization scripts:** `dev/*.py` (this directory) — each is
  self-contained and safe to run anytime (CPU-only, no GPU/model involved),
  independent of any training job that happens to be running.
