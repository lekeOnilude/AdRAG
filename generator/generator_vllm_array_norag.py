import json
import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

# Configuration
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
sampling_params = SamplingParams(
    temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512
)
llm = LLM(model=model_name)

task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
num_tasks = 8


def load_template(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


QUERY_SET = "marcov2"
RETRIEVAL_MODEL = "Qwen2.5-0.5B-bidirectional-attn-mntp-marco-passage-hard-negatives-matrioshka-reduction-2"
RUN_NAME = f"{model_name.split('/')[-1]}-NO-RAG"
PROMPT = f"./prompts/{QUERY_SET}_norag.prompt"

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

subset_data_entries = [
    entry for i, entry in enumerate(data_entries) if i % num_tasks == task_id
]


def build_prompt(query, template):

    user_content = template.format(query=query)

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


PROMPT_TEMPLATE = load_template(PROMPT)
prompts = []
for entry in tqdm(subset_data_entries, desc=f"Task {task_id}"):
    query = entry["query"]
    prompt_text = build_prompt(query, PROMPT_TEMPLATE)
    prompts.append(prompt_text)
    print(prompts)
    exit()

outputs = llm.generate(prompts, sampling_params)
assert len(outputs) == len(subset_data_entries)

output_file = os.path.join(OUTPUT_FOLDER, f"{RUN_NAME}_{task_id}.jsonl")
with open(output_file, "w", encoding="utf-8") as f:
    for output, entry in zip(outputs, subset_data_entries):
        candidate = output.outputs[0].text.strip()
        query_id = entry["query_id"]
        f.write(json.dumps({"query_id": query_id, "response": candidate}) + "\n")
