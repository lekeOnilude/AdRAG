import json

from tqdm import tqdm

# converts a tsv "text_id, title, text" to jsonl {text_id, text, title}

def tsv_to_jsonl(input_tsv, output_jsonl):
    with open(input_tsv, 'r') as tsv_file, open(output_jsonl, 'w') as jsonl_file:
        for line in tqdm(tsv_file):
            line = line.strip().split('\t')
            text_id, title, text = line
            json_data = {'docid': text_id, 'text': f"{text}", 'title': f"{title}"}
            jsonl_file.write(json.dumps(json_data) + '\n')

input_file = "/data/user_data/jmcoelho/datasets/marco/documents/corpus_firstp_2048.tsv"
output = "/data/user_data/jmcoelho/datasets/marco/documents/corpus_firstp_2048.jsonl"

tsv_to_jsonl(input_file, output)