#!/bin/bash
#SBATCH --job-name=pr_ncauses
#SBATCH --account=atmlaml
#SBATCH --partition=booster
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48

ml   Stages/2025  GCCcore/.13.3.0 Python/3.12.3
source /p/project1/profound/al-zeqri1/PFN/graphpfn/env_booster/bin/activate

cd /p/project1/profound/al-zeqri1/PFN/open-benchmark3/graphpfn/paper

echo "=== full env dump (SLURM/CUDA related) BEFORE any override ==="
env | grep -Ei 'slurm|cuda' | sort
echo "==============================================================="

# Force CUDA_VISIBLE_DEVICES to the job's GPU allocation (a value inherited from the submitting
# shell can otherwise hide the allocated GPUs).
export CUDA_VISIBLE_DEVICES="$SLURM_JOB_GPUS"

echo "=== GPU visibility diagnostics AFTER override ==="
echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "SLURM_STEP_GPUS=$SLURM_STEP_GPUS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L
echo "==================================="

# Train the pooler on 4 GPUs; the best checkpoint lands in
# dev_prior_final/output/train_pooler/<run>/pooler_checkpoint_best.pt.
torchrun --nproc_per_node=4 dev_prior_final/train_pooler.py \
    --min-molecules 1000 --max-molecules 2000 --n-sampler-workers 11 --prefetch-factor 4 \
    --n-steps 10000 --lr 0.001 --zinc-context 2000 --zinc-query 500