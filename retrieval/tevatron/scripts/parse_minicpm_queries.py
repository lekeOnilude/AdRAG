from datasets import load_dataset

from huggingface_hub import login

# Set your Hugging Face API token here
api_token = "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"

# Login to Hugging Face Hub using the provided token
login(api_token)

data_files = [f"en_{str(i).zfill(2)}.jsonl" for i in range(24)]  # Adjust range if there are more/less files

dataset = load_dataset("XBKYS/minicpm-embedding-data", data_files=data_files)

# Display the first example
print(dataset['train'][0])

for example in dataset['train']:
    query = example["query"][0]
    positive = example["query"][0]