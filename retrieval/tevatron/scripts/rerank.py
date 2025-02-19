import sys
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset
import pickle
import math

from huggingface_hub import login

SHARDS=4
n_shard = int(sys.argv[1])

login("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")

data_files = [f"en_{str(i).zfill(2)}.jsonl" for i in range(24)]  # Adjust range if there are more/less files

dataset = load_dataset("XBKYS/minicpm-embedding-data", data_files=data_files, split="train")
dataset = dataset.shard(num_shards=SHARDS, index=n_shard)

pairs = []

for example in tqdm(dataset):
    query = example["query"][1]
    positive = example["pos"][1]
    pairs.append((query, positive))

print(pairs[0])


tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')
model = model.to('cuda')
model.eval()

def batch(data, batch_size=500):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

all_scores = []
with torch.no_grad():
    for sample in tqdm(batch(pairs, 500), total=math.ceil(len(pairs) / 500)):
        inputs = tokenizer(sample, padding=True, truncation=True, return_tensors='pt', max_length=512)
        inputs = {key: value.to('cuda') for key, value in inputs.items()}
        sample_scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        all_scores.extend(sample_scores)

with open(f"rerank_minicpm_dataset_{n_shard}", 'wb') as h:
    pickle.dump(all_scores, h, protocol=pickle.HIGHEST_PROTOCOL)