import json
import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
import random

random.seed(17121998)
# Configuration
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

cases = {
    0: "Qwen/Qwen2.5-7B-Instruct", 1: "Qwen/Qwen2.5-7B-Instruct",
    2: "mistralai/Mistral-7B-Instruct-v0.3", 3: "mistralai/Mistral-7B-Instruct-v0.3",
    4: "/data/models/huggingface/meta-llama/Llama-3.1-8B-Instruct", 5: "/data/models/huggingface/meta-llama/Llama-3.1-8B-Instruct",
    6: "google/gemma-2-9b-it", 7: "google/gemma-2-9b-it"
}

model_name = cases[task_id]

print(f"Using model {model_name} to insert ads" )
tokenizer = AutoTokenizer.from_pretrained(model_name)
sampling_params = SamplingParams(
    temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=1024
)
llm = LLM(model=model_name)


PATH_TO_ANSWERS = f"/home/jmcoelho/11797_Project/generator/output/marcov2.train/Qwen2.5-0.5B-bidirectional-attn-mntp-marco-passage-hard-negatives-matrioshka-reduction-2/Qwen2.5-7B-Instruct-10-passage-RAG_{task_id}.jsonl"
PATH_TO_QUERIES = "/home/jmcoelho/11797_Project/data/marco_v2.1_qa_train/queries.tsv"

# Output folder for individual tasks
OUTPUT_FOLDER = (
    f"/home/jmcoelho/11797_Project/rewriter/output/marcov2.train/Qwen2.5-0.5B-bidirectional-attn-mntp-marco-passage-hard-negatives-matrioshka-reduction-2/Qwen2.5-7B-Instruct-10-passage-RAG/prompt1-12"
)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


qid2text = {}
with open(PATH_TO_QUERIES, 'r') as h:
    for line in h:
        qid, query = line.strip().split("\t")
        qid2text[qid] = query



def build_prompt(query, response):


    user_content = generate_prompt(query, response)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {"role": "user", "content": user_content},
    ]
    if "gemma" in model_name:
        del messages[0]

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt_text


def generate_prompt(query, response, mode='random'):
    if mode == 'fixed':
        prompt_file = "/home/jmcoelho/11797_Project/rewriter/prompts/5.prompt"
        with open(prompt_file, "r") as file:
            prompt_template = file.read()
    elif mode == 'random':
        prompt_files = [
            f"/home/jmcoelho/11797_Project/rewriter/prompts/{i}.prompt"
            for i in range(1, 13)]
        chosen_file = random.choice(prompt_files)
        with open(chosen_file, "r") as file:
            prompt_template = file.read()
    else:
        raise ValueError("Mode must be either 'fixed' or 'random'.")
    
    return prompt_template.format(query=query, response=response)


prompts = []

data_entries = []
with open(PATH_TO_ANSWERS, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data_entries.append(json.loads(line))

with open(PATH_TO_ANSWERS, 'r') as h:
    for entry in tqdm(data_entries):

        query = qid2text[entry["query_id"]]
        answer = entry["response"]

        prompt_text = build_prompt(query, answer)
        prompts.append(prompt_text)


outputs = llm.generate(prompts, sampling_params)
assert len(outputs) == len(data_entries)

output_file = os.path.join(OUTPUT_FOLDER, f"{model_name.split('/')[-1]}_{task_id}_with_ad.jsonl")
with open(output_file, "w", encoding="utf-8") as f:
    for output, entry in zip(outputs, data_entries):
        with_answer = output.outputs[0].text.strip()
        query_id = entry["query_id"]
        original = entry["response"]
        f.write(json.dumps({"query_id": query_id, "without_ad": original, "with_ad": with_answer}) + "\n")

