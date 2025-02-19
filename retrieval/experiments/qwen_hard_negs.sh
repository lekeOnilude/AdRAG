#!/bin/bash
#SBATCH --job-name=ranker-dpo
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=preempt
#SBATCH --gres=gpu:L40S:8
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-13-13,babel-13-17,babel-14-37,babel-6-9,babel-7-9,babel-3-21,babel-13-25,babel-13-1,babel-14-1,babel-12-9,babel-4-9


export NCCL_P2P_DISABLE=1

export HF_HOME=/data/group_data/cx_group/query_generation_data/hf_cache/

eval "$(conda shell.bash hook)"
conda activate qa_proj

# base_model=Qwen2.5-0.5B-bidirectional-attn-mntp
# port=$((RANDOM % (23000 - 20000 + 1) + 20000))


# model_output_name=$base_model-marco-passage-hard-negatives
# EMBEDDING_OUTPUT_DIR=/data/group_data/cx_group/query_generation_data/temporary_indexes/$model_output_name

base_model=Qwen/Qwen2.5-0.5B
port=$((RANDOM % (23000 - 20000 + 1) + 20000))


model_output_name=Qwen2.5-0.5B-marco-passage-hard-negatives
EMBEDDING_OUTPUT_DIR=/data/group_data/cx_group/query_generation_data/temporary_indexes/$model_output_name

mkdir -p $EMBEDDING_OUTPUT_DIR

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port $port --module tevatron.retriever.driver.train \
  --deepspeed tevatron/deepspeed/ds_zero3_config.json \
  --dataset_cache_dir $HF_HOME \
  --cache_dir $HF_HOME \
  --output_dir /data/user_data/jmcoelho/models/$model_output_name \
  --model_name_or_path $base_model \
  --dataset_name Tevatron/msmarco-passage-aug \
  --save_steps 20000000 \
  --query_prefix "Query: " \
  --passage_prefix "Passage" \
  --add_markers True \
  --bf16 \
  --pooling avg \
  --gradient_checkpointing \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 128 \
  --train_group_size 16 \
  --learning_rate 1e-4 \
  --query_max_len 32 \
  --passage_max_len 128 \
  --num_train_epochs 1 \
  --logging_steps 1 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 2 \
  --report_to wandb \
  --run_name $model_output_name

echo "Inference"

for shard in {0..7}; do
    (
    
    EMBEDDING_OUTPUT_FILE=$EMBEDDING_OUTPUT_DIR/corpus.marco-passage.${shard}.pkl

    if [ -f "$EMBEDDING_OUTPUT_FILE" ]; then
        echo "File $EMBEDDING_OUTPUT_FILE already exists. Skipping shard $shard."
    else
        echo "Encoding shard $shard ..."

        CUDA_VISIBLE_DEVICES=$shard python -m tevatron.retriever.driver.encode \
                --output_dir=temp \
                --bf16 \
                --model_name_or_path /data/user_data/jmcoelho/models/$model_output_name \
                --dataset_cache_dir $HF_HOME \
                --cache_dir $HF_HOME \
                --query_prefix "Query: " \
                --passage_prefix "Passage: " \
                --pooling avg \
                --normalize \
                --per_device_eval_batch_size 300 \
                --query_max_len 32 \
                --passage_max_len 128 \
                --dataset_name Tevatron/msmarco-passage-corpus \
                --add_markers True \
                --dataset_number_of_shards 8 \
                --dataset_shard_index ${shard} \
                --encode_output_path $EMBEDDING_OUTPUT_FILE
    fi
    ) &
    done
wait

CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --model_name_or_path /data/user_data/jmcoelho/models/$model_output_name \
    --bf16 \
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
    --dataset_name Tevatron/msmarco-passage \
    --dataset_split dev \
    --encode_output_path $EMBEDDING_OUTPUT_DIR/query-test-marco-passage-dev.pkl



set -f && OMP_NUM_THREADS=24 python -m tevatron.retriever.driver.search \
    --query_reps $EMBEDDING_OUTPUT_DIR/query-test-marco-passage-dev.pkl \
    --passage_reps $EMBEDDING_OUTPUT_DIR/corpus.marco-passage.*.pkl \
    --depth 1000 \
    --batch_size 128 \
    --save_text \
    --save_ranking_to $EMBEDDING_OUTPUT_DIR/run.marco-passage-dev.txt


python tevatron/src/tevatron/utils/format/convert_result_to_trec.py \
    --input $EMBEDDING_OUTPUT_DIR/run.marco-passage-dev.txt \
    --output $EMBEDDING_OUTPUT_DIR/run.marco-passage-dev.trec

rm $EMBEDDING_OUTPUT_DIR/run.marco-passage-dev.txt

eval "$(conda shell.bash hook)"
conda activate pyserini

python -m pyserini.eval.trec_eval -c -mndcg_cut.10 -mrecip_rank -mrecall.1000 msmarco-passage-dev-subset $EMBEDDING_OUTPUT_DIR/run.marco-passage-dev.trec
