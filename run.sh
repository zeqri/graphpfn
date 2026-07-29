#!/bin/bash
#SBATCH --job-name=molecule-graph-level-pooling-topology
#SBATCH --account=atmlaml
#SBATCH --partition=booster
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48

ml   Stages/2025  GCCcore/.13.3.0 Python/3.12.3
source /p/project1/profound/al-zeqri1/PFN/graphpfn/env_booster/bin/activate

# ml   Stages/2025  GCCcore/.13.3.0 Python/3.12.3
# source /p/project1/profound/al-zeqri1/PFN/graphpfn/env_booster/bin/activate

# #test graph pooling on a single dataset
# torchrun --nproc_per_node=4 paper/dev/limix_backbone_gnn_pooling_fit_test.py

#test graph pooling on multiple datasets
# torchrun --nproc_per_node=4 paper/dev/limix_backbone_gnn_pooling_multi_dataset_fit_test.py

cd /p/project1/profound/al-zeqri1/PFN/open-benchmark3/graphpfn/paper

# torchrun --nproc_per_node=4 dev/limix_backbone_gnn_pooling_multi_dataset_fit_test.py \
#     --n-molecules 3000 --n-sampler-workers 11 --prefetch-factor 4 --n-steps 10000



# torchrun --nproc_per_node=4 dev/limix_backbone_gnn_pooling_multi_dataset_varied_prior_fit_test.py \
#     --min-molecules 1000 --max-molecules 4000 --n-sampler-workers 11 --prefetch-factor 4 --n-steps 10000


# torchrun --nproc_per_node=4 dev/limix_backbone_gnn_pooling_multi_dataset_multi_agg_pooler_fit_test.py \
#     --min-molecules 1000 --max-molecules 3000 --n-sampler-workers 11 --prefetch-factor 4 --n-steps 10000


# torchrun --nproc_per_node=4 dev/limix_backbone_gnn_pooling_multi_dataset_multi_agg_pooler_fit_test.py \
#     --min-molecules 1000 --max-molecules 3000 --n-sampler-workers 11 --prefetch-factor 4 --n-steps 10000 --lr 0.003




torchrun --nproc_per_node=4 dev/limix_backbone_gnn_pooling_multi_dataset_multi_view_pooler_fit_test.py \
    --min-molecules 1000 --max-molecules 2000 --n-sampler-workers 11 --prefetch-factor 4 --n-steps 10000 --lr 0.001