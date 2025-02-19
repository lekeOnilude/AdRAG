import json


with open("./data/marco_v2.1_qa_dev/dev_v2.1.json", "r") as h:
    data = json.load(h)


query_to_gold_answers = {}
for qid in data["wellFormedAnswers"]:
    if data["wellFormedAnswers"][qid] != "[]":
        query_to_gold_answers[data["query"][qid]] = data["wellFormedAnswers"][qid]
