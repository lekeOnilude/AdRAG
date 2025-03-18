#!/bin/bash
#SBATCH --job-name=gen_data
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --array=0-20
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=64G
#SBATCH --partition=preempt
#SBATCH --time=1-00:00:00
#SBATCH --exclude=shire-1-6,shire-1-10,babel-0-[23,27,31,37],babel-1-[23,27],babel-1-31,babel-3-21,babel-4-[1,17,25,33,37],babel-6-[9,13],babel-7-[9,17],babel-11-[9,21],babel-12-9,babel-13-[1,13,17,25],babel-14-[1,21,37],babel-15-32,babel-5-19

export NCCL_P2P_DISABLE=1

SEED=$SLURM_ARRAY_TASK_ID

n_shot=5
temp=0.5
classifier="jmvcoelho/ad-classifier-v0.1"
model_dir="/data/group_data/cx_group/models/gonilude/ad_writer/sft_it0/checkpoint-953/"

python rewriter/evaluate_generate_ad.py --model-path $model_dir --n-shot 5 --clf-model-dir $classifier --n-shot $n_shot --temperature $temp