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
#SBATCH --exclude=babel-13-13,babel-13-17,babel-14-37,babel-6-9,babel-7-9,babel-3-21,babel-13-25,babel-13-1,babel-14-1,babel-12-9,babel-4-9

export HF_HOME=/data/group_data/cx_group/query_generation_data/hf_cache/

eval "$(conda shell.bash hook)"
conda activate vllm

python generator_vllm.py