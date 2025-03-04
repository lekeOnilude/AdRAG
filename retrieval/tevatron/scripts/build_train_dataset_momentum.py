from datasets import load_dataset, Dataset
import sys
import random

random.seed(17121998)

PATH_TO_NEGS = sys.argv[1]
PATH_TO_NEGS_MOMENTUM = sys.argv[2]
HF_DATASET_OUTPATH = sys.argv[3]
N_NEGS = 15

N_CURRENT, N_MOMENTUM = int((2 / 3) * N_NEGS), int((1 / 3) * N_NEGS)


print(f"Sampling {N_CURRENT} from: {PATH_TO_NEGS}")
print(f"Momentum {N_MOMENTUM} from: {PATH_TO_NEGS_MOMENTUM}")

base_dataset = load_dataset("Tevatron/msmarco-passage-aug", split="train")

qid2pos = {}
qid2text = {}
for example in base_dataset:
    if example["query_id"] not in qid2pos:
        qid2pos[example["query_id"]] = example["positive_passages"][0]["docid"]
    if example["query_id"] not in qid2text:
        qid2text[example["query_id"]] = example["query"]


corpus = load_dataset("Tevatron/msmarco-passage-corpus", split="train")
did2text = {}
for example in corpus:
    if example["docid"] not in did2text:
        did2text[example["docid"]] = {
            "text": example["text"],
            "title": example["title"],
        }

qid2negs = {}
with open(PATH_TO_NEGS, "r") as h:
    for line in h:
        qid, did, sr = line.strip().split()
        if qid not in qid2negs:
            qid2negs[qid] = {}
        qid2negs[qid][did] = float(sr)


qid2negs_momentum = {}
with open(PATH_TO_NEGS, "r") as h:
    for line in h:
        qid, did, sr = line.strip().split()
        if qid not in qid2negs_momentum:
            qid2negs_momentum[qid] = {}
        qid2negs_momentum[qid][did] = float(sr)


hf_dataset_data = {
    "query": [],
    "pos": [],
    "neg": [],
    "pos_ids": [],
    "neg_ids": [],
    "query_id": [],
}

for query_id in qid2pos:

    query = qid2text[query_id]
    positive_id = qid2pos[query_id]
    positive = (
        f"Title: {did2text[positive_id]['title']} Text: {did2text[positive_id]['text']}"
    )

    negs = qid2negs[query_id]
    assert len(negs) == 100
    negs_momentum = qid2negs_momentum[query_id]
    assert len(negs) == 100

    eligible_negatives = [
        doc_id for doc_id, score in negs.items() if doc_id != positive_id
    ]
    sampled_neg_ids = random.sample(eligible_negatives, N_CURRENT)

    eligible_negatives_momentum = [
        doc_id
        for doc_id, score in negs_momentum.items()
        if doc_id != positive_id and doc_id not in sampled_neg_ids
    ]
    sampled_negs_momentum_ids = random.sample(eligible_negatives_momentum, N_MOMENTUM)

    sampled_neg_texts = [
        f"Title: {did2text[x]['title']} Text: {did2text[x]['text']}"
        for x in sampled_neg_ids + sampled_negs_momentum_ids
    ]

    hf_dataset_data["query"].append(["prompt", query])
    hf_dataset_data["pos"].append(["prompt", positive])
    hf_dataset_data["neg"].append(["prompt"] + sampled_neg_texts)
    hf_dataset_data["query_id"].append([query_id])
    hf_dataset_data["pos_ids"].append([positive_id])
    hf_dataset_data["neg_ids"].append(sampled_neg_ids)

dataset = Dataset.from_dict(hf_dataset_data)
print(dataset)
dataset.save_to_disk(HF_DATASET_OUTPATH)
