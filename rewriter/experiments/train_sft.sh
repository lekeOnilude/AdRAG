#!/bin/bash
#SBATCH --job-name=train_dpo
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH -e slurm_logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:8
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-8-13,babel-13-13,babel-13-17,babel-14-37,babel-6-9,babel-7-9,babel-3-21,babel-13-25,babel-13-1,babel-14-1,babel-12-9,babel-4-9


export HF_HOME=/data/group_data/cx_group/
export NCCL_P2P_DISABLE=1
export HF_DATASETS_CACHE=/data/group_data/cx_group/
export WANDB_API_KEY=b30c01d6e7de72ad731317f2acb2721ae6869749
export WANDB_PROJECT=CodeCMU
export WANDB_ENTITY=gonilude

deepspeed --include localhost:0,1,2,3,4,5,6,7  rewriter/train_models/stf.py