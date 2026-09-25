# FS-Mol with the MolPFN pooler

This folder **meta-trains the MolPFN pooler** on the real training assays of [FS-Mol](https://github.com/microsoft/FS-Mol) and **evaluates it on the 157 held-out test tasks**, following the FS-Mol protocol: support sizes 16/32/64/128/256, 10 stratified resamples per task and size, metric = delta-AUPRC.

Only the pooler is trained; the LimiX-16M backbone stays frozen. Meta-training does not start from scratch: it continues from the pooler pretrained on the synthetic prior (see the [top-level README](../../README.md#1-train-the-pooler)). The folder also contains three embeddings-free baselines (LimiX, TabICL v2, TabPFN v3) that run on each molecule's ECFP fingerprint.

## What you need

| Item | Where it goes | How to get it |
|---|---|---|
| MolPFN environment | — | The `venv` from the top-level [Setup](../../README.md#setup) |
| LimiX-16M backbone | `paper/checkpoints/LimiX-16M.ckpt` | Top-level [Setup](../../README.md#setup), step 2 |
| Pretrained MolPFN pooler | `checkpoints/pooler_checkpoint_best.pt` | Included. Training starts from it, and the evaluators load it when `--pooler-checkpoint` is not given |
| FS-Mol data | `datasets/fs-mol/` | See [Setup](#setup) below |
| TabICL v2 / TabPFN v3 checkpoints (baselines only) | `paper/checkpoints/{tabicl,tabpfnv3}/` | Top-level [TabICL](../../README.md#3-tabicl-baseline) and [TabPFN v3](../../README.md#4-tabpfn-v3-baseline) sections |

All paths are relative to the repository root. Unless stated otherwise, run every command below from this folder (`paper/fs_mol/`) with the MolPFN `venv` active.

## Setup

**1. Download the FS-Mol task files and task lists.** The task files are a ~4.9 GB tarball on figshare; the task lists and target metadata live in the FS-Mol GitHub repository. Run this from the **repository root**; it creates `datasets/fs-mol/`:

```bash
mkdir -p datasets && cd datasets

# task files: fs-mol/{train,valid,test}/, one .jsonl.gz per assay
curl -L --fail -C - -# -o fsmol.tar https://ndownloader.figshare.com/files/31345321
echo "0ad8f73db6bb8415b023538d514138dc  fsmol.tar" | md5sum -c -    # must print: fsmol.tar: OK
tar -xf fsmol.tar

# task lists and target metadata
git clone --depth 1 https://github.com/microsoft/FS-Mol fsmol_repo_tmp
cp fsmol_repo_tmp/datasets/fsmol-0.1.json fsmol_repo_tmp/datasets/entire_train_set.json fs-mol/
cp fsmol_repo_tmp/datasets/targets/target_info.csv fs-mol/
rm -rf fsmol_repo_tmp

cd ..
```

You should now have:

```
datasets/fs-mol/
├── train/                  # one .jsonl.gz per training assay
├── valid/                  # 40 validation assays
├── test/                   # 157 test assays
├── fsmol-0.1.json          # the training tasks used in the FS-Mol paper
├── entire_train_set.json
└── target_info.csv         # enzyme classes, used for the per-EC breakdown
```

The tarball can be deleted after extracting. By default only the training tasks listed in `fsmol-0.1.json` are used, which keeps results comparable to the FS-Mol baselines.

**2. Pick a feature recipe.** Each recipe has its own trainer and evaluator, and a checkpoint must be evaluated with the recipe it was trained with:

| Recipe | Atom features | Fingerprint | Trainer / evaluator |
|---|---|---|---|
| `native_features` | FS-Mol's 32-col `node_features` | no | [`train_…_native_features.py`](train_graphpfn_pooler_on_fsmol_meta_train_native_features.py) / [`eval_…_native_features.py`](evaluation/eval_graphpfn_pooler_on_fsmol_test_native_features.py) |
| `native_features_plus_fp` | FS-Mol's 32-col `node_features` | yes | [`train_…_native_features_plus_fp.py`](train_graphpfn_pooler_on_fsmol_meta_train_native_features_plus_fp.py) / [`eval_…_native_plus_fingerprint.py`](evaluation/eval_graphpfn_pooler_on_fsmol_test_native_plus_fingerprint.py) |
| `x9_plus_extra_plus_fp` | 40 cols: 9 MoleculeNet-style + 31 RDKit | yes | [`train_…_x9_plus_extra_plus_fp.py`](train_graphpfn_pooler_on_fsmol_meta_train_x9_plus_extra_plus_fp.py) / [`eval_…_x9_features.py`](evaluation/eval_graphpfn_pooler_on_fsmol_test_x9_features.py) |

The fingerprint recipes splice each molecule's ECFP fingerprint onto its pooled graph representation. The recipe definitions are in [`fsmol_core/recipes.py`](fsmol_core/recipes.py).

## 1. Meta-train the pooler

```bash
python train_graphpfn_pooler_on_fsmol_meta_train_native_features.py
python train_graphpfn_pooler_on_fsmol_meta_train_native_features_plus_fp.py --fp-fold 128
python train_graphpfn_pooler_on_fsmol_meta_train_x9_plus_extra_plus_fp.py --fp-fold 128
```

**Multiple GPUs:** launch with `torch.distributed.run` instead of plain `python`; the flags stay the same:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
    train_graphpfn_pooler_on_fsmol_meta_train_x9_plus_extra_plus_fp.py --fp-fold 128
```

Each optimizer step accumulates gradients over 20 episodes. An episode samples one training task, splits it into a stratified support and query set, and trains on the query loss. Every 200 steps the EMA weights are scored on the 40 validation tasks, and the best delta-AUPRC at support size 16 is kept.

| Argument | Default | Meaning |
|---|---|---|
| `--pooler-checkpoint` | `checkpoints/pooler_checkpoint_best.pt` | Checkpoint to start from. Must have the same pooler architecture |
| `--fp-fold` | 0 (no folding) | Fold the 2048-bit fingerprint to this width (fingerprint recipes only). **Evaluation must use the same value** |
| `--lr` | 0.0001 | Peak learning rate (cosine schedule with warmup) |
| `--n-steps` | 20000 | Total optimizer steps |
| `--grad-accum-steps` | 20 | Episodes per optimizer step |
| `--support-sizes` | `16,32,64,128` | Support sizes to sample episodes at |
| `--max-query` | 128 | Query molecules per episode. Lower it if training runs out of memory |
| `--eval-every` / `--patience` | 200 / 25 | Steps between validation rounds / rounds without improvement before stopping early |
| `--output-dir` | see below | Where checkpoints are written |

Run any trainer with `--help` for the full list.

**Outputs** are written to `paper/fs_mol/output/fsmol_meta_train_<recipe>/lr_<lr>_supp_<sizes>[_fp_<fold>]/`:

- `pooler_checkpoint_best.pt`: the checkpoint with the best validation delta-AUPRC. **Use this one for evaluation.**
- `pooler_checkpoint.pt`: the latest state. If you rerun with the same arguments, training resumes from it automatically.
- `training_log.jsonl`: one line per validation round.

A resumed run ignores `--pooler-checkpoint`. To start from a different checkpoint, use a new output directory (a different `--lr`, `--support-sizes` or `--fp-fold` changes the directory name, or pass `--output-dir`).

## 2. Evaluate

Each evaluator scores one checkpoint on all 157 test tasks. For the fingerprint recipes, pass the **same `--fp-fold` as in training** (the evaluator's default is 512) and add `--no-baseline`, which skips the fingerprint-free pass that the checkpoint was not trained for.

```bash
CKPT=$PWD/output

# x9 + extra + fingerprint
python evaluation/eval_graphpfn_pooler_on_fsmol_test_x9_features.py \
    --per-mol-features fingerprint --fp-fold 128 --no-baseline \
    --pooler-checkpoint $CKPT/fsmol_meta_train_x9_plus_extra_plus_fp/lr_0.0001_supp_16-32-64-128_fp_128/pooler_checkpoint_best.pt \
    --output-json evaluation/output/x9_fp128.json

# native + fingerprint
python evaluation/eval_graphpfn_pooler_on_fsmol_test_native_plus_fingerprint.py \
    --per-mol-features fingerprint --fp-fold 128 --no-baseline \
    --pooler-checkpoint $CKPT/fsmol_meta_train_native_features_plus_fp/lr_0.0001_supp_16-32-64-128_fp_128/pooler_checkpoint_best.pt \
    --output-json evaluation/output/native_fp128.json

# native, no fingerprint (no --fp-fold, no --no-baseline)
python evaluation/eval_graphpfn_pooler_on_fsmol_test_native_features.py \
    --pooler-checkpoint $CKPT/fsmol_meta_train_native_features/lr_0.0001_supp_16-32-64-128/pooler_checkpoint_best.pt \
    --output-json evaluation/output/native.json
```

Pass a different `--output-json` for each run. Otherwise the output name depends only on the script, and a second run overwrites the first.

| Argument | Default | Meaning |
|---|---|---|
| `--pooler-checkpoint` | `checkpoints/pooler_checkpoint_best.pt` | Pooler `.pt` to evaluate |
| `--fp-fold` | 512 | Fingerprint width. Must match training |
| `--per-mol-features` | `fingerprint` | `fingerprint`, `descriptors` or `both` (fingerprint recipes only) |
| `--no-baseline` | off | Skip the fingerprint-free pass (fingerprint recipes only) |
| `--support-sizes` | `16,32,64,128,256` | Support sizes to evaluate |
| `--num-runs` | 10 | Resamples per task and support size |
| `--no-ema` | off | Use raw instead of EMA pooler weights |
| `--output-json` | `evaluation/[output/]<script>.json` | Where to write metrics |

`--tasks`, `--limit-tasks` and `--max-query` make the result **not comparable** to the FS-Mol paper.

## 3. Read the results

Print the per-support-size table from a result JSON:

```python
import json
d = json.load(open("evaluation/output/x9_fp128.json"))
S = d["summary"][d["variants"][0]]
print("support  n_tasks   delta-AUPRC        ROC-AUC          AP")
for size in sorted(S, key=int):
    a = S[size]["all_enzymes"]
    print(f"{size:>7} {S[size]['n_tasks']:>8}  "
          f"{a['delta_auprc']['mean']:+.4f}±{a['delta_auprc']['sem']:.4f}  "
          f"{a['roc_auc']['mean']:.4f}±{a['roc_auc']['sem']:.4f}  "
          f"{a['ap']['mean']:.4f}±{a['ap']['sem']:.4f}")
```

The `±` is the standard error across tasks. Only 43 test tasks have enough molecules for support size 256, so that row averages over fewer tasks.

## 4. Baselines: LimiX, TabICL v2, TabPFN v3

The baselines run a frozen tabular foundation model directly on each molecule's ECFP fingerprint, with no pooler and no graph. They use the same 157 test tasks, splits and metrics as the pooler evaluators, and their result JSONs have the same layout, so the snippet in [section 3](#3-read-the-results) prints their tables too.

| Baseline | Script | Default checkpoint | Environment |
|---|---|---|---|
| LimiX | [`eval_limix_backbone_on_fsmol_test_fingerprints.py`](evaluation/LimiX/eval_limix_backbone_on_fsmol_test_fingerprints.py) | `paper/checkpoints/LimiX-16M.ckpt` | MolPFN `venv` |
| TabICL v2 | [`eval_tabiclv2_on_fsmol_test_fingerprints.py`](evaluation/TabICL/eval_tabiclv2_on_fsmol_test_fingerprints.py) | `paper/checkpoints/tabicl/tabicl-classifier-v2-20260212.ckpt` | MolPFN `venv` |
| TabPFN v3 | [`eval_tabpfnv3_on_fsmol_test_fingerprints.py`](evaluation/TabPFNv3/eval_tabpfnv3_on_fsmol_test_fingerprints.py) | `paper/checkpoints/tabpfnv3/tabpfn-v3-classifier-v3_default.ckpt` | separate `tabpfn_env` |

TabPFN v3 needs its own environment because it requires `torch>=2.5`, and its weights need a one-time license acceptance. Set up the checkpoints and `tabpfn_env` as described in the top-level [TabICL](../../README.md#3-tabicl-baseline) and [TabPFN v3](../../README.md#4-tabpfn-v3-baseline) sections.

**LimiX and TabICL v2** run in the MolPFN `venv`:

```bash
python evaluation/LimiX/eval_limix_backbone_on_fsmol_test_fingerprints.py     # needs a GPU
python evaluation/TabICL/eval_tabiclv2_on_fsmol_test_fingerprints.py
```

**TabPFN v3** runs in `tabpfn_env`:

```bash
deactivate 2>/dev/null           # leave the MolPFN venv if it is active
source ../../tabpfn_env/bin/activate

python evaluation/TabPFNv3/eval_tabpfnv3_on_fsmol_test_fingerprints.py
```

Each script writes its result JSON to an `output/` folder next to it, for example `evaluation/TabICL/output/eval_tabiclv2_on_fsmol_test_fingerprints.json`. Use `--output-json` to choose another path.

**Common options** (shared by all three baselines):

| Argument | Default | Meaning |
|---|---|---|
| `--per-mol-features` | `fingerprint` | `fingerprint`, `descriptors` or `both` |
| `--fp-fold` | 512 | Fold the 2048-bit fingerprint to this width (`0` = no folding) |
| `--support-sizes` | `16,32,64,128,256` | Support sizes to evaluate |
| `--num-runs` | 10 | Resamples per task and support size |
| `--fsmol-dir` | `datasets/fs-mol` | FS-Mol data folder |
| `--backbone-checkpoint` / `--inference-config` | `paper/checkpoints/LimiX-16M.ckpt` / `paper/vendor/limix/config/cls_default_16M_retrieval.json` | LimiX only |
| `--checkpoint-dir` | `paper/checkpoints/{tabicl,tabpfnv3}` | TabICL v2 and TabPFN v3 only |

Run any script with `--help` for the full list.

## Folder layout

```
paper/fs_mol/
├── fsmol_core/                                        # shared code: data, recipes, training loop, evaluation, metrics
├── train_graphpfn_pooler_on_fsmol_meta_train_<recipe>.py   # one trainer per recipe
├── evaluation/
│   ├── eval_graphpfn_pooler_on_fsmol_test_<recipe>.py      # one evaluator per recipe
│   ├── merge_<recipe>_shards.py                            # combine sharded evaluation results
│   ├── LimiX/  TabICL/  TabPFNv3/                          # fingerprint baselines
│   └── output/                                             # result JSONs
└── output/                                            # meta-training checkpoints (generated)
```
