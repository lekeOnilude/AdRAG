#!/bin/bash
#SBATCH --job-name=inference_array
#SBATCH --array=0-7
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00

eval "$(conda shell.bash hook)"
conda activate vllm

export HF_HOME=/data/group_data/cx_group/query_generation_data/hf_cache/
export NCCL_P2P_DISABLE=1

echo $SLURM_ARRAY_TASK_ID

mkdir -p /tmp/$SLURM_ARRAY_TASK_ID/.outlines
export OUTLINES_CACHE_DIR=/tmp/$SLURM_ARRAY_TASK_ID/.outlines

df -h /tmp

python generate_items.py