#!/bin/bash
#SBATCH --job-name=inference_bm25
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-13-13,babel-13-17,babel-14-37,babel-6-9,babel-7-9,babel-3-21,babel-13-25,babel-13-1,babel-14-1,babel-12-9,babel-4-9




eval "$(conda shell.bash hook)"
conda activate pyserini

export PYSERINI_CACHE=/data/group_data/cx_group/query_generation_data/hf_cache/


python -m pyserini.search.lucene \
  --threads 48 --batch-size 128 \
  --index msmarco-v2.1-doc-segmented \
  --topics /home/jmcoelho/11797_Project/data/marco_v2.1_qa_dev/queries.tsv \
  --output /data/group_data/cx_group/query_generation_data/temporary_indexes/bm25/marco_v2_segmented/run.marcov2.txt \
  --bm25 --hits 100

