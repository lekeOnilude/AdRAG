#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
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

file_split=$1

shard1=$(printf "%02d" $(( $1 * 2 )))
shard2=$(printf "%02d" $(( $1 * 2 + 1 )))

echo $shard1
echo $shard2

mkdir -p $EMBEDDING_OUTPUT_DIR


for shard in $shard1 $shard2; do
    EMBEDDING_OUTPUT_FILE=$EMBEDDING_OUTPUT_DIR/corpus.marco.v2.passage.segmented.${shard}.pkl

    if [ -f "$EMBEDDING_OUTPUT_FILE" ]; then
        echo "File $EMBEDDING_OUTPUT_FILE already exists. Skipping shard $shard."
    else
        echo "Encoding shard $shard ..."

        python -m tevatron.retriever.driver.encode \
            --output_dir=temp \
            --bf16 \
            --dim_reduction_factor 2 \
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
            --dataset_path /data/group_data/cx_group/temporary/marco_v2.1_segmented/msmarco_v2.1_doc_segmented/msmarco_v2.1_doc_segmented_${shard}.json \
            --add_markers True \
            --encode_output_path $EMBEDDING_OUTPUT_FILE
    fi
done