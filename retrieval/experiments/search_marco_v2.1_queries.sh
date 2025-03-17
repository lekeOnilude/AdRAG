#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:8
#SBATCH --cpus-per-task=1
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-13-13,babel-13-17,babel-14-37,babel-6-9,babel-7-9,babel-3-21,babel-13-25,babel-13-1,babel-14-1,babel-12-9,babel-4-9


export NCCL_P2P_DISABLE=1

export HF_HOME=/data/group_data/cx_group/query_generation_data/hf_cache/

eval "$(conda shell.bash hook)"
conda activate qa_proj

base_model=Qwen2.5-0.5B-bidirectional-attn-mntp
port=$((RANDOM % (23000 - 20000 + 1) + 20000))
model_output_name=$base_model-marco-passage-hard-negatives-matrioshka-reduction-2
EMBEDDING_OUTPUT_DIR=/data/group_data/cx_group/query_generation_data/temporary_indexes/$model_output_name/marco_v2_segmented


CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --model_name_or_path /data/user_data/jmcoelho/models/$model_output_name \
    --bf16 \
    --dim_reduction_factor 2 \
    --pooling avg \
    --normalize \
    --dataset_cache_dir $HF_HOME \
    --cache_dir $HF_HOME \
    --query_prefix "Query: " \
    --passage_prefix "Passage: " \
    --encode_is_query \
    --per_device_eval_batch_size 300 \
    --query_max_len 32 \
    --passage_max_len 128 \
    --dataset_path /home/jmcoelho/11797_Project/data/marco_v2.1_qa_train/queries_with_answer_and_bing_passages.jsonl \
    --encode_output_path $EMBEDDING_OUTPUT_DIR/query-marcov2-train.pkl



set -f && OMP_NUM_THREADS=24 python -m tevatron.retriever.driver.search \
    --query_reps $EMBEDDING_OUTPUT_DIR/query-marcov2-train.pkl \
    --passage_reps $EMBEDDING_OUTPUT_DIR/corpus.marco.v2.passage.segmented.*.pkl \
    --depth 100 \
    --batch_size 128 \
    --save_text \
    --save_ranking_to $EMBEDDING_OUTPUT_DIR/run.marcov2.train.txt

