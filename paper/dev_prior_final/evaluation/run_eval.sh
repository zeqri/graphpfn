#!/bin/bash
#SBATCH --job-name=graphpfn-limix-model-uncertainty
#SBATCH --account=atmlaml
#SBATCH --partition=booster
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --array=0-17
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# --limix-seed sweep (model uncertainty) over every evaluation script in this directory, for both
# embedding models. One SLURM array task = one (embedding model, dataset) pair on one GPU; all SEEDS
# for that pair run sequentially. Same fixed --seed / --n-ensemble / --max-train / --test-chunk-size
# per dataset as the earlier uncertainty sweeps.
#
# Submit FROM THIS DIRECTORY (sbatch runs a copy of this file, so the submit dir is used to find the
# scripts). No paths are baked in -- pass them through the environment:
#   POOLER_CHECKPOINT  pooler checkpoint (.pt) to evaluate                        [required]
#   MOLECULENET_ROOT   override the MoleculeNet root (default <repo>/datasets/moleculenet)   [optional]
#   ZINC_ROOT          override the ZINC root        (default <repo>/datasets/zinc)          [optional]
#   AQSOL_ROOT         override the AQSOL root       (default <repo>/datasets/aqsol)         [optional]
#   VENV               virtualenv to activate                                      [optional]
# Datasets (<repo>/datasets), the backbone (<paper>/checkpoints/LimiX-16M.ckpt) and embeddings
# (<repo>/embeddings/<model>/<dataset>) default to paths relative to the scripts.
# Results: outputs/<model>/<dataset>/eval.limixseed<seed>.json
#
# Array index -> (model, dataset): 0-8 = Molbert x DATASETS, 9-17 = MolDeBERTa x DATASETS, with
# DATASETS = esol freesolv lipo bace bbbp clintox sider zinc aqsol.
#
#   POOLER_CHECKPOINT=... VENV=... sbatch runs.sh
#   POOLER_CHECKPOINT=... VENV=... sbatch --array=7,8,16,17 runs.sh     # zinc + aqsol
#   POOLER_CHECKPOINT=... SLURM_ARRAY_TASK_ID=0 bash runs.sh            # interactive
set -euo pipefail

: "${POOLER_CHECKPOINT:?set POOLER_CHECKPOINT}"
: "${SLURM_ARRAY_TASK_ID:?set SLURM_ARRAY_TASK_ID (0-17)}"

module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3
if [[ -n "${VENV:-}" ]]; then
    source "${VENV}/bin/activate"
fi

cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"

EMBEDDING_MODELS=(Molbert MolDeBERTa)
DATASETS=(esol freesolv lipo bace bbbp clintox sider zinc aqsol)
SEEDS=(1 2 3 4 5)

declare -A CMDS
CMDS[esol]="eval_moleculenet_regression.py --dataset esol --seed 0 --n-ensemble 1"
CMDS[freesolv]="eval_moleculenet_regression.py --dataset freesolv --seed 0 --n-ensemble 1"
CMDS[lipo]="eval_moleculenet_regression.py --dataset lipo --seed 0 --n-ensemble 10 --max-train 2000"
CMDS[bace]="eval_bace.py"
CMDS[bbbp]="eval_bbbp.py"
CMDS[clintox]="eval_clintox.py"
CMDS[sider]="eval_sider.py --tasks all"
CMDS[zinc]="eval_zinc.py --max-train 2000 --n-ensemble 10 --seed 0"
CMDS[aqsol]="eval_aqsol.py --max-train 2000 --n-ensemble 10 --seed 0 --test-chunk-size 500"

emb=${EMBEDDING_MODELS[$((SLURM_ARRAY_TASK_ID / ${#DATASETS[@]}))]}
ds=${DATASETS[$((SLURM_ARRAY_TASK_ID % ${#DATASETS[@]}))]}

# Dataset root overrides (only passed when set; otherwise the scripts use <repo>/datasets/...).
ROOT_ARGS=()
case "${ds}" in
    zinc)  [[ -n "${ZINC_ROOT:-}" ]] && ROOT_ARGS=(--zinc-root "${ZINC_ROOT}") ;;
    aqsol) [[ -n "${AQSOL_ROOT:-}" ]] && ROOT_ARGS=(--aqsol-root "${AQSOL_ROOT}") ;;
    *)     [[ -n "${MOLECULENET_ROOT:-}" ]] && ROOT_ARGS=(--moleculenet-root "${MOLECULENET_ROOT}") ;;
esac

mkdir -p "outputs/${emb}/${ds}"

for limix_seed in "${SEEDS[@]}"; do
    echo "[task ${SLURM_ARRAY_TASK_ID}] embedding=${emb} dataset=${ds} limix-seed=${limix_seed}"
    python ${CMDS[$ds]} \
        --device cuda \
        "${ROOT_ARGS[@]}" \
        --pooler-checkpoint "${POOLER_CHECKPOINT}" \
        --embedding-model "${emb}" \
        --limix-seed "${limix_seed}" \
        --output-json "outputs/${emb}/${ds}/eval.limixseed${limix_seed}.json"
done
