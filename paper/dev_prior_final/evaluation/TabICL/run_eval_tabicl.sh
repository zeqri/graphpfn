#!/bin/bash
# Embeddings-only TabICL baseline (eval_tabicl.py) over every dataset, for both embedding models, with
# a --tabicl-seed sweep. Everything runs sequentially on one GPU: for each embedding model,
# each dataset, each seed. Per-dataset --max-train / --n-ensemble defaults live in
# eval_tabicl.py and match run_eval_all_seeds.sh.
#
# TabICL checkpoints are not in this repo. Download them once, from the repository root, into
# <paper>/checkpoints/tabicl/ (see eval_tabicl.py):
#   python -c "from huggingface_hub import hf_hub_download as d; [d('jingang/TabICL', f, local_dir='paper/checkpoints/tabicl') \
#       for f in ('tabicl-classifier-v2-20260212.ckpt', 'tabicl-regressor-v2-20260212.ckpt')]"
#
# Environment:
#   TABICL_CHECKPOINT_DIR  directory with both .ckpt files (default: <paper>/checkpoints/tabicl)  [optional]
#   EMBEDDING_MODELS   space-separated models (default: "Molbert MolDeBERTa")        [optional]
#   DATASETS           space-separated datasets (default: all nine, see below)       [optional]
#   SEEDS              space-separated --tabicl-seed values (default: "1 2 3 4 5")   [optional]
#   OUTPUT_ROOT        results root (default: outputs)                           [optional]
#   OVERWRITE=1        re-run seeds whose JSON already exists                     [optional]
#   MOLECULENET_ROOT / ZINC_ROOT / AQSOL_ROOT   dataset root overrides            [optional]
#   VENV               virtualenv to activate                                     [optional]
# Results: ${OUTPUT_ROOT}/<model>/<dataset>/tabicl.seed<seed>.json
#
#   bash run_eval_tabicl.sh
#   DATASETS="zinc aqsol" bash run_eval_tabicl.sh     # zinc + aqsol
set -euo pipefail

if [[ -n "${VENV:-}" ]]; then
    source "${VENV}/bin/activate"
fi
export HF_HUB_OFFLINE=1

cd "$(dirname "${BASH_SOURCE[0]}")"

OUTPUT_ROOT=${OUTPUT_ROOT:-outputs}
OVERWRITE=${OVERWRITE:-0}
TABICL_CHECKPOINT_DIR=${TABICL_CHECKPOINT_DIR:-../../../checkpoints/tabicl}

# Checked up front: with HF_HUB_OFFLINE=1 TabICL cannot fetch a missing checkpoint.
for ckpt in tabicl-classifier-v2-20260212.ckpt tabicl-regressor-v2-20260212.ckpt; do
    [[ -f "${TABICL_CHECKPOINT_DIR}/${ckpt}" ]] || {
        echo "missing ${TABICL_CHECKPOINT_DIR}/${ckpt} -- download the TabICL checkpoints first (see the header)" >&2
        exit 1
    }
done

read -r -a EMBEDDING_MODELS <<< "${EMBEDDING_MODELS:-Molbert MolDeBERTa}"
read -r -a DATASETS <<< "${DATASETS:-esol freesolv lipo bace bbbp clintox sider zinc aqsol}"
read -r -a SEEDS <<< "${SEEDS:-1 2 3 4 5}"

declare -A EXTRA_ARGS
EXTRA_ARGS[sider]="--tasks all"

for emb in "${EMBEDDING_MODELS[@]}"; do
    for ds in "${DATASETS[@]}"; do
        # Dataset root / checkpoint overrides (only passed when set).
        ROOT_ARGS=()
        case "${ds}" in
            zinc)  [[ -n "${ZINC_ROOT:-}" ]] && ROOT_ARGS=(--zinc-root "${ZINC_ROOT}") ;;
            aqsol) [[ -n "${AQSOL_ROOT:-}" ]] && ROOT_ARGS=(--aqsol-root "${AQSOL_ROOT}") ;;
            *)     [[ -n "${MOLECULENET_ROOT:-}" ]] && ROOT_ARGS=(--moleculenet-root "${MOLECULENET_ROOT}") ;;
        esac
        ROOT_ARGS+=(--checkpoint-dir "${TABICL_CHECKPOINT_DIR}")

        out_dir="${OUTPUT_ROOT}/${emb}/${ds}"
        mkdir -p "${out_dir}"

        for tabicl_seed in "${SEEDS[@]}"; do
            out_json="${out_dir}/tabicl.seed${tabicl_seed}.json"
            if [[ -f "${out_json}" && "${OVERWRITE}" != 1 ]]; then
                echo "${out_json} exists, skipping (OVERWRITE=1 to re-run)"
                continue
            fi
            echo "embedding=${emb} dataset=${ds} tabicl-seed=${tabicl_seed}"
            python eval_tabicl.py --dataset "${ds}" ${EXTRA_ARGS[$ds]:-} \
                --device cuda \
                "${ROOT_ARGS[@]}" \
                --embedding-model "${emb}" \
                --tabicl-seed "${tabicl_seed}" \
                --output-json "${out_json}"
        done
    done
done
