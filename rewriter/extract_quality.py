import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from tqdm import tqdm


# Load a suitable LLM (e.g., Flan-T5)
model_name = "Qwen/Qwen2.5-7B-Instruct"  # Or another suitable model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")


def extract_qualities(text, advertisement):
    """Extracts qualities related to an advertisement from a given text."""
    prompt = f"""
    Identify the qualities of the {advertisement} being discussed in the following text.
    Return a comma-separated list only of these qualities with no additional information.

    Text: {text}
    """

    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()


# Load data from valid.jsonl
rows = {}
with open("preprocess_data/valid.jsonl", "r") as f:
    for line in f:
        row = json.loads(line)
        rows[row["id"]] = row

# Process each row and extract qualities
for id, row in tqdm(rows.items()):
    extracted_qualities = extract_qualities(row["adv_sen_span"], row["advertisement"])
    row["qualities"] = extracted_qualities  # Add the extracted qualities to the row

# Save the updated data (optional)
with open("preprocess_data/valid_with_qualities.jsonl", "w") as outfile:
    for id, row in rows.items():
        json.dump(row, outfile)
        outfile.write("\n")

