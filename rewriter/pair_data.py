import json
from tqdm import tqdm
import numpy as np


def parse_labels_to_dict(jsonl_path):
    query_id_to_label = {}
    with open(jsonl_path, "r") as jsonl_file:
        for line in tqdm(jsonl_file, desc="Processing labels"):
            data = json.loads(line)
            if data["id"] not in query_id_to_label:                
                query_id_to_label[data["id"]] = {}
            query_id_to_label[data["id"]]["label"] = int(data["label"])
            query_id_to_label[data["id"]]["advertisement"] = data["advertisement"]
            query_id_to_label[data["id"]]["sen_span"] = data["sen_span"]
    return query_id_to_label

def parse_responses_to_dict(jsonl_path):
    query_id_to_response = {}
    with open(jsonl_path, "r") as jsonl_file:
        for line in tqdm(jsonl_file, desc="Processing responses"):
            data = json.loads(line)
            if data["id"] not in query_id_to_response:
                query_id_to_response[data["id"]] = {}
            query_id_to_response[data["id"]]["response"] = data["response"]
            query_id_to_response[data["id"]]["meta_topic"] = data["meta_topic"]
            query_id_to_response[data["id"]]["query"] = data["query"]
    return query_id_to_response



valid_labels_jsonl_path = "./data/subtask-2/responses-validation-labels.jsonl"
valid_responses_jsonl_path = "./data/subtask-2/responses-validation.jsonl"
valid_label = parse_labels_to_dict(valid_labels_jsonl_path)



def get_ids_w_and_wo_ads(labels_dict):
    ids_w_ads = set()
    ids_wo_ads = set()

    for query_id, label_dict in labels_dict.items():
        id_num = query_id.split("-")[:-1]
        id_num = "-".join(id_num)
        label = label_dict["label"]
        if label == 1:
            ids_w_ads.add(id_num)
        else:
            ids_wo_ads.add(id_num)
    
    common_ids = ids_w_ads.intersection(ids_wo_ads)
    return common_ids



def preprocess_data(common_ids, valid_label, valid_responses):
    with open("./preprocess_data/valid.jsonl", "w") as f:
        for id in common_ids:
            result = {}
            id_w_ads = id + "-A"
            id_wo_ads = id + "-N"

            text_wo_ads = valid_responses[id_wo_ads]["response"]
            text_w_ads = valid_responses[id_w_ads]["response"]
            
            advertistment = valid_label[id_w_ads]["advertisement"]
            query = valid_responses[id_w_ads]["query"]
            meta_topic = valid_responses[id_w_ads]["meta_topic"]
            sent_span = valid_label[id_w_ads]["sen_span"].split(",")
            start_span = int(sent_span[0][1:].strip())
            end_span = int(sent_span[1][:-1].strip())


            result['id'] = id_wo_ads
            result['text'] = text_wo_ads
            result['advertisement'] = advertistment
            result['query'] = query
            result['meta_topic'] = meta_topic
            result['target'] = text_w_ads
            result['adv_sen_span'] = text_w_ads[start_span:end_span]
            f.write(json.dumps(result) + "\n")
            
            

valid_responses = parse_responses_to_dict(valid_responses_jsonl_path)

preprocess_data(get_ids_w_and_wo_ads(valid_label), valid_label, valid_responses)

