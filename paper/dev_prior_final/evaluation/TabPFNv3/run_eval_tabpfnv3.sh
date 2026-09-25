#!/bin/bash
# Embeddings-only TabPFN v3 baseline (eval_tabpfnv3.py) over every dataset, for both embedding models,
# with a --tabpfn-seed sweep. Everything runs sequentially on one GPU: for each embedding model,
# each dataset, each seed. Per-dataset --max-train / --n-ensemble defaults live in
# eval_tabpfnv3.py and match the TabICL baseline.
#
# TabPFN v3 checkpoints are not in this repo. They are read from <paper>/checkpoints/tabpfnv3/
# (tabpfn-v3-classifier-v3_default.ckpt / tabpfn-v3-regressor-v3_default.ckpt); a missing one is
# downloaded there by tabpfn on first use, after a one-time license acceptance (see eval_tabpfnv3.py).
#
# Environment:
#   TABPFN_CHECKPOINT_DIR  directory with both .ckpt files (default: <paper>/checkpoints/tabpfnv3)  [optional]
#   EMBEDDING_MODELS   space-separated models (default: "Molbert MolDeBERTa")        [optional]
#   DATASETS           space-separated datasets (default: all nine, see below)       [optional]
#   SEEDS              space-separated --tabpfn-seed values (default: "1 2 3 4 5")   [optional]
#   OUTPUT_ROOT        results root (default: outputs)                           [optional]
#   OVERWRITE=1        re-run seeds whose JSON already exists                     [optional]
#   DATASETS_ROOT      root holding moleculenet/, zinc/, aqsol/                   [optional]
#   VENV               virtualenv to activate (the separate tabpfn environment)   [optional]
# Results: ${OUTPUT_ROOT}/<model>/<dataset>/tabpfnv3.seed<seed>.json
#
#   bash run_eval_tabpfnv3.sh
#   DATASETS="zinc aqsol" bash run_eval_tabpfnv3.sh     # zinc + aqsol
set -euo pipefail

if [[ -n "${VENV:-}" ]]; then
    source "${VENV}/bin/activate"
fi

cd "$(dirname "${BASH_SOURCE[0]}")"

OUTPUT_ROOT=${OUTPUT_ROOT:-outputs}
OVERWRITE=${OVERWRITE:-0}
TABPFN_CHECKPOINT_DIR=${TABPFN_CHECKPOINT_DIR:-../../../checkpoints/tabpfnv3}

read -r -a EMBEDDING_MODELS <<< "${EMBEDDING_MODELS:-Molbert MolDeBERTa}"
read -r -a DATASETS <<< "${DATASETS:-esol freesolv lipo bace bbbp clintox sider zinc aqsol}"
read -r -a SEEDS <<< "${SEEDS:-1 2 3 4 5}"

declare -A EXTRA_ARGS
EXTRA_ARGS[sider]="--tasks all"

# Dataset root / checkpoint overrides (only passed when set).
ROOT_ARGS=(--checkpoint-dir "${TABPFN_CHECKPOINT_DIR}")
[[ -n "${DATASETS_ROOT:-}" ]] && ROOT_ARGS+=(--datasets-root "${DATASETS_ROOT}")

for emb in "${EMBEDDING_MODELS[@]}"; do
    for ds in "${DATASETS[@]}"; do
        out_dir="${OUTPUT_ROOT}/${emb}/${ds}"
        mkdir -p "${out_dir}"

        for tabpfn_seed in "${SEEDS[@]}"; do
            out_json="${out_dir}/tabpfnv3.seed${tabpfn_seed}.json"
            if [[ -f "${out_json}" && "${OVERWRITE}" != 1 ]]; then
                echo "${out_json} exists, skipping (OVERWRITE=1 to re-run)"
                continue
            fi
            echo "embedding=${emb} dataset=${ds} tabpfn-seed=${tabpfn_seed}"
            python eval_tabpfnv3.py --dataset "${ds}" ${EXTRA_ARGS[$ds]:-} \
                --device cuda \
                "${ROOT_ARGS[@]}" \
                --embedding-model "${emb}" \
                --tabpfn-seed "${tabpfn_seed}" \
                --output-json "${out_json}"
        done
    done
done
