import json
import glob
from tqdm import tqdm

MODEL_NAME = "Qwen2.5-0.5B-bidirectional-attn-mntp-marco-passage-hard-negatives-matrioshka-reduction-2"
QUERIES = "marcov2" #touche

PATH_RUN = f"/data/group_data/cx_group/query_generation_data/temporary_indexes/{MODEL_NAME}/marco_v2_segmented/run.{QUERIES}.txt"

QUERIES_PATHS = {
    "touche": "/home/jmcoelho/11797_Project/data/subtask-1/queries.jsonl",
    "marcov2": "/home/jmcoelho/11797_Project/data/marco_v2.1_qa_dev/queries_with_answer_and_bing_passages.jsonl"
}


PATH_CORPUS = "/data/group_data/cx_group/temporary/marco_v2.1_segmented/msmarco_v2.1_doc_segmented/msmarco_v2.1_doc_segmented_*.json"
PATH_OUT = f"./output/{QUERIES}/{MODEL_NAME}.jsonl"

qid2pids = {}
with open(PATH_RUN, "r") as f:
    for line in f:
        qid, pid, _ = line.strip().split()
        if qid not in qid2pids:
            qid2pids[qid] = []
        qid2pids[qid].append(pid)

queries = {}
answers = {}
with open(QUERIES_PATHS[QUERIES], "r") as f:
    for line in f:
        query = json.loads(line)
        queries[query["query_id"]] = query["query"]

        if "answers" in query:
            answers[query["query_id"]] = query["answers"]

corpus = {}
for file in tqdm(glob.glob(PATH_CORPUS)):
    with open(file, "r") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["docid"]] = f"Title: {doc["title"]}. Text: {doc["segment"]}."

with open(PATH_OUT, "w") as f:
    for qid in qid2pids:
        query_text = queries[qid]
        passages_text = [corpus[pid] for pid in qid2pids[qid]]
        data = {
            "query_id": qid,
            "query": query_text,
            "passages": passages_text
        }

        if answers != {}:
            data["answers"] = answers[qid]

        f.write(json.dumps(data) + "\n")
