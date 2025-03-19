import json

# Q_PATH = "./data/marco_v2.1_qa_dev/dev_v2.1.json"
Q_PATH = "/data/group_data/cx_group/temporary/train_v2.1.json"

with open(Q_PATH, "r") as h:
    data = json.load(h)

# set of 150k queries. this will be used to train the classifier.


# qid_to_gold_answers = {}
# qid_to_text = {}
# qid_to_bing_results = {}
# for qid in data["wellFormedAnswers"]:
#     if data["wellFormedAnswers"][qid] != "[]":
#         qid_to_gold_answers[qid] = data["wellFormedAnswers"][qid]
#         qid_to_text[qid] = data["query"][qid]
#         qid_to_bing_results[qid] = [p["passage_text"] for p in data["passages"][qid]]

# with open(
#     "./data/marco_v2.1_qa_train/queries_with_answer_and_bing_passages.jsonl", "w"
# ) as f:
#     for qid in qid_to_text:
#         data = {
#             "query_id": qid,
#             "query": qid_to_text[qid],
#             "answers": qid_to_gold_answers[qid],
#             "passages": qid_to_bing_results[qid],
#         }
#         f.write(json.dumps(data) + "\n")


# with open("./data/marco_v2.1_qa_train/queries.tsv", "w") as f:
#     for qid in qid_to_text:
#         f.write("\t".join([qid, qid_to_text[qid]]) + "\n")



# set of 150k queries different from the one above. can be used to train the ad rewriter.

qid_to_text = {}
qid_to_bing_results = {}
for qid in data["wellFormedAnswers"]:
    if data["wellFormedAnswers"][qid] == "[]":
        qid_to_text[qid] = data["query"][qid]
        qid_to_bing_results[qid] = [p["passage_text"] for p in data["passages"][qid]]
        if len(qid_to_text) == 150000:
            break

with open(
    "./data/marco_v2.1_qa_train_no_ans/queries_without_answer_and_bing_passages_150k.jsonl", "w"
) as f:
    for qid in qid_to_text:
        data = {
            "query_id": qid,
            "query": qid_to_text[qid],
            "passages": qid_to_bing_results[qid],
        }
        f.write(json.dumps(data) + "\n")


with open("./data/marco_v2.1_qa_train_no_ans/queries.tsv", "w") as f:
    for qid in qid_to_text:
        f.write("\t".join([qid, qid_to_text[qid]]) + "\n")

