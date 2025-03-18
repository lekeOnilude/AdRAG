import json
import os
from tqdm import tqdm


def load_jsonl_data(file_path):
    """Load a JSONL file into a list of dictionaries."""
    data_dict = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading sub-task 1 data"):
            data = json.loads(line.strip())
            id = data['query']["id"]
            data_dict[id] = []
            for ad in data['advertisements']:
                if ad:
                    data_dict[id].append({
                        "advertisement" : ad['item'],
                        "meta_topic" : ad['type'],
                        "qualities": ad['qualities']
                    })
    return data_dict


def load_jsonl_response(file_path):
    """Load a JSONL file into a list of dictionaries."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading QA response data"):
            data.append(json.loads(line.strip()))
    return data


data = load_jsonl_data('data/subtask-1/touche-task4-2025-segments.jsonl')
response = load_jsonl_response('preprocess_data/touche.Qwen2.5-7B-Instruct-10-passage-RAG.jsonl')

output_data = []

for res in tqdm(response, desc="Merging QA response data and Sub-task 1 data"):
    res_wo_ad_id = res['query_id']
    res_wo_ad = res['response']
    ads = data[res_wo_ad_id]
    for ad in ads:
        output = {}
        output['id'] = res_wo_ad_id
        output['text'] = res_wo_ad
        output['advertisement'] = ad['advertisement']
        output['meta_topic'] = ad['meta_topic']
        output['qualities'] = ad['qualities']
        output_data.append(output)


OUTPUT_FOLDER = 'preprocess_data'
output_file = os.path.join(OUTPUT_FOLDER, "data.jsonl")
with open(output_file, "w") as outfile:
    for result in tqdm(output_data, desc="Writing output file"):
        json.dump(result, outfile)
        outfile.write("\n")


