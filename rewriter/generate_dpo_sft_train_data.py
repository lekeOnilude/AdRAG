import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

classifier_model_path = "jmvcoelho/ad-classifier-v0.1"
tokenizer = AutoTokenizer.from_pretrained(classifier_model_path)
model = AutoModelForSequenceClassification.from_pretrained(classifier_model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def get_ad_likelihood(passages):
    inputs = tokenizer(
        passages, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        ad_probs = probs[:, 1].cpu().tolist()
    return ad_probs

def process_jsonl_file_with_batching(input_file, output_file, batch_size=300):
    max_likelihoods = []
    min_likelihoods = []
    
    all_data = []
    all_passages = []
    
    with open(input_file, 'r') as fin:
        for line in fin:
            data = json.loads(line)
            assert len(data["answers_with_ad"]) == 5
            all_data.append(data)
            all_passages.extend(data["answers_with_ad"])
    
    all_ad_probs = []
    total_batches = (len(all_passages) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(all_passages), batch_size), total=total_batches, desc="Processing batches"):
        batch_passages = all_passages[i:i+batch_size]
        batch_probs = get_ad_likelihood(batch_passages)
        all_ad_probs.extend(batch_probs)
    
    with open(output_file, 'w') as fout:
        for i, data in enumerate(tqdm(all_data)):
            start_idx = i * 5
            ad_probs = all_ad_probs[start_idx:start_idx + 5]
            
            max_likelihoods.append(max(ad_probs))
            min_likelihoods.append(min(ad_probs))
            
            data["ad_likelihood"] = ad_probs
            fout.write(json.dumps(data) + '\n')
    
    return min_likelihoods, max_likelihoods


def build_filtered_datasets(jsonl_file, items_file, output_min_file, output_min_max_file):
    items_dict = {}
    with open(items_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            items_dict[entry['query_id']] = entry['item']
    
    min_dataset = []
    min_max_dataset = []
    
    with open(jsonl_file, 'r') as f:
        for line in tqdm(f, desc="Building datasets"):
            data = json.loads(line)
            query_id = data['query_id']
            
            likelihoods = data['ad_likelihood']
            answers = data['answers_with_ad']
            response = data['response']
            
            if min(likelihoods) < 0.5 and max(likelihoods) > 0.5:
                min_idx = likelihoods.index(min(likelihoods))
                max_idx = likelihoods.index(max(likelihoods))
                
                best_answer = answers[min_idx]
                worst_answer = answers[max_idx]
                
                item = items_dict.get(query_id, {})
                
                min_entry = {
                    "response": response,
                    "item": item,
                    "best_answer": best_answer
                }
                
                min_max_entry = {
                    "response": response,
                    "item": item,
                    "best_answer": best_answer,
                    "worst_answer": worst_answer
                }
                
                min_dataset.append(min_entry)
                min_max_dataset.append(min_max_entry)
    
    with open(output_min_file, 'w') as f:
        for entry in min_dataset:
            f.write(json.dumps(entry) + '\n')
    
    with open(output_min_max_file, 'w') as f:
        for entry in min_max_dataset:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Created dataset with best answers only: {len(min_dataset)} entries")
    print(f"Created dataset with best and worst answers: {len(min_max_dataset)} entries")
    
    return min_dataset, min_max_dataset

def plot_kde(d1, d2, output_file):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(d1, label="Minimum Ad Likelihood", shade=True, color="#FDB515")
    sns.kdeplot(d2, label="Maximum Ad Likelihood", shade=True, color="#008F91")
    plt.legend()
    plt.xlabel("Ad Likelihood")
    plt.ylabel("Density")
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


PATH_IN = "/home/jmcoelho/11797_Project/rewriter/output/marcov2.train.set2/answers_with_ads/generated_ads_Qwen2.5-7B-Instruct_temp_1.0.jsonl"
PATH_OUT= "/home/jmcoelho/11797_Project/rewriter/output/marcov2.train.set2/answers_with_ads/generated_ads_Qwen2.5-7B-Instruct_temp_1.0_with_ad_likelihood.jsonl"
min_likelihoods, max_likelihoods = process_jsonl_file_with_batching(PATH_IN, PATH_OUT)

PATH_KDE_OUT = "/home/jmcoelho/11797_Project/rewriter/output/marcov2.train.set2/answers_with_ads/kde.pdf"
plot_kde(min_likelihoods, max_likelihoods, PATH_KDE_OUT)

PATH_ITEMS = "/home/jmcoelho/11797_Project/rewriter/output/marcov2.train.set2/items/Qwen2.5-7B-Instruct-10-passage-RAG_with_item.jsonl"
PATH_OUT_STF = "/home/jmcoelho/11797_Project/rewriter/output/marcov2.train.set2/answers_with_ads/generated_ads_Qwen2.5-7B-Instruct_temp_1.0_stf.jsonl"
PATH_OUT_DPO = "/home/jmcoelho/11797_Project/rewriter/output/marcov2.train.set2/answers_with_ads/generated_ads_Qwen2.5-7B-Instruct_temp_1.0_dpo.jsonl"
build_filtered_datasets(PATH_OUT, PATH_ITEMS, PATH_OUT_STF, PATH_OUT_DPO)