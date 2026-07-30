#!/bin/bash
#SBATCH --job-name=molecule-graph-level-pooling-topology
#SBATCH --account=atmlaml
#SBATCH --partition=booster
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48

ml   Stages/2025  GCCcore/.13.3.0 Python/3.12.3
source /p/project1/profound/al-zeqri1/PFN/graphpfn/env_booster/bin/activate


cd /p/project1/profound/al-zeqri1/PFN/open-benchmark3/graphpfn/paper

# python dev/diagnose_conv_type_size_r2.py \
#     --checkpoint-path dev/output/limix_backbone_gnn_pooling_multi_dataset_multi_agg_pooler_fit_test/molecules_1000_3000_n_steps_10000/pooler_checkpoint.pt \
#     --min-molecules 500 --max-molecules 3000 --n-samples-per-conv-type 25 --n-workers 8


python dev/diagnose_conv_type_size_r2_multi_view_pooler.py \
    --checkpoint-path dev/output/limix_backbone_gnn_pooling_multi_dataset_multi_view_pooler_fit_test/molecules_1000_2000_n_steps_10000_lr_0.001/pooler_checkpoint.pt \
    --min-molecules 500 --max-molecules 2000 --n-samples-per-conv-type 25 --n-workers 8