import json

from tqdm import tqdm

# converts a tsv "text_id, title, text" to jsonl {text_id, text, title}

def tsv_to_jsonl(input_tsv, output_jsonl):
    with open(input_tsv, 'r') as tsv_file, open(output_jsonl, 'w') as jsonl_file:
        for line in tqdm(tsv_file):
            line = line.strip().split('\t')
            qid, text = line
            json_data = {'query_id': qid, 'query': f"{text}"}
            jsonl_file.write(json.dumps(json_data) + '\n')

input_file = "/data/user_data/jmcoelho/datasets/marco/documents/dev.query.txt"
output = "/data/user_data/jmcoelho/datasets/marco/documents/dev.query.jsonl"

tsv_to_jsonl(input_file, output)