#!/bin/bash
# Regenerate every pretrained-molecule embedding the evaluation uses.
#
# Inputs are read from <graphpfn>/datasets/ (raw CSVs + scaffold splits from
# datasets/make_scaffold_splits.py; each scaffold-split dataset is refused unless its raw CSV
# matches the sha256 in split/split_info.json). All outputs land in
# <graphpfn>/embeddings/{MolDeBERTa,Molbert}/<name>/
#   <name>_embeddings_{train,valid,test}.npy + <name>_meta.json
#
# Pretrained models are read from <graphpfn>/embeddings/checkpoints/ (see the README):
#   Mole-BERT/                   clone of https://github.com/junxia97/Mole-BERT
#                                (loader.py, model.py, model_gin/Mole-BERT.pth)
#   MolDeBERTa-base-123M-mtr/    download of SaeedLab/MolDeBERTa-base-123M-mtr
#
# Activate the Python environment before running, or set VENV to its directory.
# Override the model locations with MOLEBERT_DIR / MOLEBERT_CHECKPOINT / MOLDEBERTA_MODEL.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
REPO_DIR=$(realpath ../..)
CKPT_DIR=${REPO_DIR}/embeddings/checkpoints

export MOLEBERT_DIR=${MOLEBERT_DIR:-${CKPT_DIR}/Mole-BERT}
export MOLEBERT_CHECKPOINT=${MOLEBERT_CHECKPOINT:-${MOLEBERT_DIR}/model_gin/Mole-BERT.pth}
MOLDEBERTA_MODEL=${MOLDEBERTA_MODEL:-${CKPT_DIR}/MolDeBERTa-base-123M-mtr}
DEVICE=${DEVICE:-cpu}

for f in "${MOLEBERT_DIR}/loader.py" "${MOLEBERT_DIR}/model.py" "${MOLEBERT_CHECKPOINT}" \
         "${MOLDEBERTA_MODEL}/config.json" "${MOLDEBERTA_MODEL}/model.safetensors"; do
    [[ -f "${f}" ]] || { echo "missing ${f} -- download the models first (README: 'Regenerate the embeddings')" >&2; exit 1; }
done

if [[ -n "${VENV:-}" ]]; then
    set +u  # venv activation reads unset variables
    source "${VENV}/bin/activate"
    set -u
fi

# The models are local, so nothing is fetched from the Hugging Face Hub.
export HF_HUB_OFFLINE=1

# =========================================================================
# MolDeBERTa
# =========================================================================

# MoleculeNet-family datasets (bace, bbbp, clintox, esol, freesolv, lipo, sider) + aqsol.
python moldeberta_encode_moleculenet.py --model "${MOLDEBERTA_MODEL}" --device "${DEVICE}"

# ZINC-12k (separate script: train/val/test are independent PyG datasets, not CSV-row subsets).
python moldeberta_encode_zinc.py --model "${MOLDEBERTA_MODEL}" --device "${DEVICE}"

# =========================================================================
# Mole-BERT (single script, all datasets: bace, bbbp, clintox, esol, freesolv,
# lipo, sider, aqsol, zinc)
# =========================================================================

python molebert_encode_all.py --device "${DEVICE}"
