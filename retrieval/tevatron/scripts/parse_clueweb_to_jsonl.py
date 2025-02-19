import json
from tqdm import tqdm
import pickle


SEQ_ID = 0
id_mapper = {}

def parse_jsonl(input_file, output_file):
    global SEQ_ID
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in tqdm(infile):
            data = json.loads(line)
            new_data = {
                'docid': SEQ_ID,
                'text': data['contents'],
                'title': ''
            }
            id_mapper[data['id']] = SEQ_ID
            SEQ_ID += 1

            json.dump(new_data, outfile)
            outfile.write('\n')


input_file1 = '/data/jcoelho/clueweb22_indexes/raw/full_corpus_en0000.jsonl'  # Replace with your input file path
input_file2 = '/data/jcoelho/clueweb22_indexes/raw/full_corpus_en0001.jsonl'  # Replace with your input file path
input_file3 = '/data/jcoelho/clueweb22_indexes/raw/full_corpus_en0002.jsonl'  # Replace with your input file path
input_file4 = '/data/jcoelho/clueweb22_indexes/raw/full_corpus_en0003.jsonl'  # Replace with your input file path
input_file5 = '/data/jcoelho/clueweb22_indexes/raw/full_corpus_en0004.jsonl'  # Replace with your input file path
output_file1 = '/data/jcoelho/clueweb22_indexes/raw/full_corpus_en0000.seqid.jsonl'  # Replace with your output file path
output_file2 = '/data/jcoelho/clueweb22_indexes/raw/full_corpus_en0001.seqid.jsonl'  # Replace with your output file path
output_file3 = '/data/jcoelho/clueweb22_indexes/raw/full_corpus_en0002.seqid.jsonl'  # Replace with your output file path
output_file4 = '/data/jcoelho/clueweb22_indexes/raw/full_corpus_en0003.seqid.jsonl'  # Replace with your output file path
output_file5 = '/data/jcoelho/clueweb22_indexes/raw/full_corpus_en0004.seqid.jsonl'  # Replace with your output file path

parse_jsonl(input_file1, output_file1)
parse_jsonl(input_file2, output_file2)
parse_jsonl(input_file3, output_file3)
parse_jsonl(input_file4, output_file4)
parse_jsonl(input_file5, output_file5)

with open("/data/jcoelho/clueweb22_indexes/raw/id_mapper.pkl", 'wb') as h:
    pickle.dump(id_mapper, h, protocol=pickle.HIGHEST_PROTOCOL)