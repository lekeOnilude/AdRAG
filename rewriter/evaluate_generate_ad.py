
import json
from vllm import LLM, SamplingParams
import argparse
from transformers import AutoTokenizer
import os
from predict_rewriter import load_classifier, predict_label
from sklearn.metrics import classification_report
from tqdm import tqdm




def load_model(model_name):
    
    # Load the default sampling parameters from the model.
    llm = LLM(model=model_name, download_dir='/data/group_data/cx_group/')
    sampling_params = llm.get_default_sampling_params()
    sampling_params.max_tokens = 1024
    sampling_params.temperature =1.0
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return llm, sampling_params, tokenizer

def create_prompt(row, tokenizer):
    user_content = f"""Rewrite the following text to make it more engaging and evocative. 
    In your rewrite, subtly incorporate an implict advertisement strategy for '{row['advertisement']}'within the {row['meta_topic']} sector, 
    highlighting the following product qualities {row['qualities']}. 
    Ensure the ad blends naturally into the text without being overly promotional. 
    Return only the revised text with the embedded ad.

    Original Text: '{row['text']}'
    """

    messages = [
        {"role": "user", "content": user_content},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt_text

def read_jsonl(file_path):
    rows = []
    with open(file_path, 'r') as f:
        for line in f:
            row = json.loads(line)
            rows.append(row)
    
    return rows

def generate_text(prompts, llm, sampling_params):
    llm_outputs = llm.generate(prompts, sampling_params)
    texts = [llm_output.outputs[0].text for llm_output in llm_outputs]
    return texts



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Predict labels for generated responses.')
    parser.add_argument(
        '--model-path', 
        type=str, 
        default="/data/group_data/cx_group/models/gonilude/ad_writer/sft_it0/checkpoint-953/",
        help='dir to the FT model')
    parser.add_argument(
        '--clf-model-dir', 
        type=str, 
        default="jmvcoelho/ad-classifier-v0.1",
        help='classifier model directory on huggingface')
    parser.add_argument(
        '--file-path', 
        type=str, 
        default="preprocess_data/data.jsonl",
        help='input data file path')
    
    args = parser.parse_args()
    
    file_path = args.file_path
    model_path = args.model_path
    clf_model_dir = args.clf_model_dir
    

    rows = read_jsonl(file_path)

    print("Loading model and tokenizer...")
    llm, sampling_params, tokenizer = load_model(model_path)

    prompts = []
    for row in rows:
        prompts.append(create_prompt(row, tokenizer))

    print(f"Generating text with {len(prompts)} prompts...")
    generated_text = generate_text(prompts, llm, sampling_params)

    print(f"Loading classifier {clf_model_dir}...")
    clf, clf_tokenizer = load_classifier(clf_model_dir)

    predictions = []
    for text in tqdm(generated_text, desc="Predicting labels"):
        predictions.append(predict_label(text, clf_tokenizer, clf))

    labels = [1] * len(predictions)
    
    report = classification_report(labels, predictions, output_dict=True, zero_division=0) 
    result = {
        "report": report,
        "accuracy": report['accuracy'],
        "f1-score": report['weighted avg']['f1-score'],
    }
    print(result)