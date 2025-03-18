#!/bin/bash
#SBATCH --job-name=evaluate_sft
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH -e slurm_logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=babel-8-13,babel-13-13,babel-13-17,babel-14-37,babel-6-9,babel-7-9,babel-3-21,babel-13-25,babel-13-1,babel-14-1,babel-12-9,babel-4-9

export NCCL_P2P_DISABLE=1

SEED=$SLURM_ARRAY_TASK_ID

n_shot=5
temp=1
classifier="jmvcoelho/ad-classifier-v0.1"
model_dir="/data/group_data/cx_group/models/gonilude/ad_writer/sft_it0/checkpoint-953/"

python rewriter/evaluate_generate_ad.py --model-path $model_dir --n-shot 5 --clf-model-dir $classifier --n-shot $n_shot --temperature $temp