import json
from tqdm import tqdm

MODEL_NAME = "BM25"
PATH_QUERIES = (
    "/home/jmcoelho/11797_Project/data/subtask-1/touche-task4-2025-segments.jsonl"
)
PATH_OUT = f"./output/{MODEL_NAME}.jsonl"

qid2texts = {}
with open(PATH_QUERIES, "r") as f, open(PATH_OUT, "w") as out:
    for line in f:
        entry = json.loads(line)
        qid = entry["query"]["id"]
        query_text = entry["query"]["text"]
        passages_text = [f"Title: {d["doc"]["title"]}. Text: {d["doc"]["segment"]}." for d in entry["candidates"]]
        data = {"query_id": qid, "query": query_text, "passages": passages_text}
        out.write(json.dumps(data) + "\n")
