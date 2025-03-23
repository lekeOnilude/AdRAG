#!/bin/bash
#SBATCH --job-name=inference_single
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00

eval "$(conda shell.bash hook)"
conda activate vllm

export HF_HOME=/data/group_data/cx_group/query_generation_data/hf_cache/
export NCCL_P2P_DISABLE=1

n_shot=0
temp=1
classifier="jmvcoelho/ad-classifier-v0.1"
#model_dir="/data/user_data/jmcoelho/models/ad_writer/dpo_it0/"
model_dir="Qwen/Qwen2.5-7B-Instruct"

python rewriter/evaluate_generate_ad.py --model-path $model_dir --clf-model-dir $classifier --n-shot $n_shot --temperature $temp