import json


QUERIES = "marcov2"  # touche

QUERIES_PATH = "/home/jmcoelho/11797_Project/data/marco_v2.1_qa_dev/queries_with_answer_and_bing_passages.jsonl"


PATH_OUT = f"./output/{QUERIES}/bing.jsonl"


queries = {}
answers = {}
passages = {}
with open(QUERIES_PATH, "r") as f:
    for line in f:
        query = json.loads(line)
        queries[query["query_id"]] = query["query"]
        answers[query["query_id"]] = query["answers"]
        passages[query["query_id"]] = query["passages"]


with open(PATH_OUT, "w") as f:
    for qid in queries:
        query_text = queries[qid]
        passages_text = passages[qid]
        answers_text = answers[qid]
        data = {
            "query_id": qid,
            "query": query_text,
            "passages": passages_text,
            "answers": answers_text,
        }

        f.write(json.dumps(data) + "\n")
