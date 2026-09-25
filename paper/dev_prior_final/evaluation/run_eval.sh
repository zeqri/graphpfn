#!/bin/bash
# --limix-seed sweep (model uncertainty) over every evaluation script in this directory, for both
# embedding models. Everything runs sequentially on one GPU: for each embedding model, each dataset,
# each seed. Same fixed --seed / --n-ensemble / --max-train per dataset as the
# earlier uncertainty sweeps.
#
# No paths are baked in -- pass them through the environment:
#   POOLER_CHECKPOINT  pooler checkpoint (.pt) to evaluate                        [required]
#   EMBEDDING_MODELS   space-separated models (default: "Molbert MolDeBERTa")      [optional]
#   DATASETS           space-separated datasets (default: all nine, see below)     [optional]
#   SEEDS              space-separated --limix-seed values (default: "1 2 3 4 5")  [optional]
#   OUTPUT_ROOT        results root (default: outputs)                             [optional]
#   OVERWRITE=1        re-run seeds whose JSON already exists                       [optional]
#   MOLECULENET_ROOT   override the MoleculeNet root (default <repo>/datasets/moleculenet)   [optional]
#   ZINC_ROOT          override the ZINC root        (default <repo>/datasets/zinc)          [optional]
#   AQSOL_ROOT         override the AQSOL root       (default <repo>/datasets/aqsol)         [optional]
#   VENV               virtualenv to activate                                      [optional]
# Datasets (<repo>/datasets), the backbone (<paper>/checkpoints/LimiX-16M.ckpt) and embeddings
# (<repo>/embeddings/<model>/<dataset>) default to paths relative to the scripts.
# Results: ${OUTPUT_ROOT}/<model>/<dataset>/eval.limixseed<seed>.json
#
#   POOLER_CHECKPOINT=... bash run_eval.sh
#   POOLER_CHECKPOINT=... DATASETS="zinc aqsol" bash run_eval.sh     # zinc + aqsol
set -euo pipefail

: "${POOLER_CHECKPOINT:?set POOLER_CHECKPOINT}"
# Resolved before the cd below, so a path relative to the caller's directory still works.
POOLER_CHECKPOINT=$(realpath "${POOLER_CHECKPOINT}")

if [[ -n "${VENV:-}" ]]; then
    source "${VENV}/bin/activate"
fi

cd "$(dirname "${BASH_SOURCE[0]}")"

OUTPUT_ROOT=${OUTPUT_ROOT:-outputs}
OVERWRITE=${OVERWRITE:-0}

read -r -a EMBEDDING_MODELS <<< "${EMBEDDING_MODELS:-Molbert MolDeBERTa}"
read -r -a DATASETS <<< "${DATASETS:-esol freesolv lipo bace bbbp clintox sider zinc aqsol}"
read -r -a SEEDS <<< "${SEEDS:-1 2 3 4 5}"

declare -A CMDS
CMDS[esol]="eval_moleculenet_regression.py --dataset esol --seed 0 --n-ensemble 1"
CMDS[freesolv]="eval_moleculenet_regression.py --dataset freesolv --seed 0 --n-ensemble 1"
CMDS[lipo]="eval_moleculenet_regression.py --dataset lipo --seed 0 --n-ensemble 10 --max-train 2000"
CMDS[bace]="eval_bace.py"
CMDS[bbbp]="eval_bbbp.py"
CMDS[clintox]="eval_clintox.py"
CMDS[sider]="eval_sider.py --tasks all"
CMDS[zinc]="eval_zinc.py --max-train 2000 --n-ensemble 10 --seed 0"
CMDS[aqsol]="eval_aqsol.py --max-train 2000 --n-ensemble 10 --seed 0"

for emb in "${EMBEDDING_MODELS[@]}"; do
    for ds in "${DATASETS[@]}"; do
        # Dataset root overrides (only passed when set; otherwise the scripts use <repo>/datasets/...).
        ROOT_ARGS=()
        case "${ds}" in
            zinc)  [[ -n "${ZINC_ROOT:-}" ]] && ROOT_ARGS=(--zinc-root "${ZINC_ROOT}") ;;
            aqsol) [[ -n "${AQSOL_ROOT:-}" ]] && ROOT_ARGS=(--aqsol-root "${AQSOL_ROOT}") ;;
            *)     [[ -n "${MOLECULENET_ROOT:-}" ]] && ROOT_ARGS=(--moleculenet-root "${MOLECULENET_ROOT}") ;;
        esac

        out_dir="${OUTPUT_ROOT}/${emb}/${ds}"
        mkdir -p "${out_dir}"

        for limix_seed in "${SEEDS[@]}"; do
            out_json="${out_dir}/eval.limixseed${limix_seed}.json"
            if [[ -f "${out_json}" && "${OVERWRITE}" != 1 ]]; then
                echo "${out_json} exists, skipping (OVERWRITE=1 to re-run)"
                continue
            fi
            echo "embedding=${emb} dataset=${ds} limix-seed=${limix_seed}"
            python ${CMDS[$ds]} \
                --device cuda \
                "${ROOT_ARGS[@]}" \
                --pooler-checkpoint "${POOLER_CHECKPOINT}" \
                --embedding-model "${emb}" \
                --limix-seed "${limix_seed}" \
                --output-json "${out_json}"
        done
    done
done
