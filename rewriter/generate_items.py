from pydantic import BaseModel
import os
import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import json
from tqdm import tqdm
from pydantic import create_model
from typing import Literal

from collections import Counter


jsonl_file = '/home/jmcoelho/11797_Project/data/subtask-1/touche-task4-2025-segments.jsonl'
type_counter = Counter()

with open(jsonl_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        advertisements = data.get('advertisements', [])
        
        for ad in advertisements:
            if ad is not None and 'type' in ad:
                type_counter[ad['type'].strip()] += 1

unique_types = list(type_counter.keys())

ItemDescription = create_model(
    'ItemDescription',
    item=(str, ...),
    type=(Literal[*unique_types], ...),  #Dynamic Literal
    qualities=(str, ...)
)

json_schema = ItemDescription.model_json_schema()
guided_decoding_params = GuidedDecodingParams(json=json_schema)


task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

model_name = "Qwen/Qwen2.5-7B-Instruct"

print(f"Using model {model_name} to insert ads" )
tokenizer = AutoTokenizer.from_pretrained(model_name)
sampling_params = SamplingParams(
    guided_decoding=guided_decoding_params, temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=1024
)
llm = LLM(model=model_name)


prompt = """You are given an answer to some user query. Your task is to analyze the content of the answer and generate a single advertisement candidate that would be relevant to append to this content.

1. Review the provided answer text carefully
2. Identify the main topic and purpose of the content
3. Generate exactly one product or service that would be relevant to users interested in this content
4. Output your advertisement candidate in this specific JSON format:

{{"item": "[product/service name]", "type": "[product/service type]", "qualities": "[brief description of key qualities]"}}

Example:
Given an answer about VR fitness games and workouts, the advertisement candidate was:
{{"item": "Zwift", "type": "service", "qualities": "immersive, gamified indoor cycling & running workout"}}

Important notes:
- Do not include an item or service that is already mentioned in the provided answer. Avoid simple variations of mentioned items as well.
- Only output the JSON-formatted advertisement candidate. Do not include explanations or any other text.

Provided answer:

{answer}

Item:"""

PATH_TO_ANSWERS=f"/home/jmcoelho/11797_Project/generator/output/marcov2.train.set2/Qwen2.5-0.5B-bidirectional-attn-mntp-marco-passage-hard-negatives-matrioshka-reduction-2/Qwen2.5-7B-Instruct-10-passage-RAG_{task_id}.jsonl"

def build_prompt(answer):


    user_content = prompt.format(answer=answer)

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



prompts = []

data_entries = []
with open(PATH_TO_ANSWERS, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data_entries.append(json.loads(line))

with open(PATH_TO_ANSWERS, 'r') as h:
    for entry in tqdm(data_entries):

        answer = entry["response"]
        query_id = entry["query_id"]
        prompt_text = build_prompt(answer)
        prompts.append(prompt_text)

outputs = llm.generate(prompts, sampling_params)
assert len(outputs) == len(data_entries)

output_file = os.path.join("/home/jmcoelho/11797_Project/rewriter/output/marcov2.train.set2/items", f"Qwen2.5-7B-Instruct-10-passage-RAG_{task_id}_item.jsonl")
with open(output_file, "w", encoding="utf-8") as f:
    for output, entry in zip(outputs, data_entries):
        item = output.outputs[0].text.strip()
        query_id = entry["query_id"]
        response = entry["response"]
        f.write(json.dumps({"query_id": query_id, "response": response, "item": item}) + "\n")

