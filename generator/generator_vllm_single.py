import json
import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

# Configuration
model_name = "Qwen/Qwen2.5-72B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
sampling_params = SamplingParams(
    temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512
)
llm = LLM(model=model_name, tensor_parallel_size=8)


QUERY_SET = "touche"
RETRIEVAL_MODEL = "Qwen2.5-0.5B-bidirectional-attn-mntp-marco-passage-hard-negatives-matrioshka-reduction-2"
RUN_NAME = "Qwen2.5-72B-Instruct-10-passage-RAG"

INPUT_FILE = (
    f"/home/jmcoelho/11797_Project/retrieval/output/{QUERY_SET}/{RETRIEVAL_MODEL}.jsonl"
)
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

# Output folder for individual tasks
OUTPUT_FOLDER = (
    f"/home/jmcoelho/11797_Project/generator/output/{QUERY_SET}/{RETRIEVAL_MODEL}"
)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


data_entries = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data_entries.append(json.loads(line))


def build_prompt(query, passages):

    nl = "\n"
    user_content = f"""Answer the following web query, given the context. 
    Context: {nl.join(passages[:10])}.
    Query: {query}.
    Reply only with an 'well formed answer': a human-like complete sentence, brief and direct to the point.
    """

    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. For this interaction, you will be answering a web search query.",
        },
        {"role": "user", "content": user_content},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt_text


prompts = []
for entry in tqdm(data_entries):
    query = entry["query"]
    passages = entry.get("passages", [])
    prompt_text = build_prompt(query, passages)
    prompts.append(prompt_text)

outputs = llm.generate(prompts, sampling_params)
assert len(outputs) == len(data_entries)

output_file = os.path.join(OUTPUT_FOLDER, f"{RUN_NAME}_{1}.jsonl")
with open(output_file, "w", encoding="utf-8") as f:
    for output, entry in zip(outputs, data_entries):
        candidate = output.outputs[0].text.strip()
        query_id = entry["query_id"]
        f.write(json.dumps({"query_id": query_id, "response": candidate}) + "\n")
