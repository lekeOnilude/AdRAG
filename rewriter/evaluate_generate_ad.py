
import json
from vllm import LLM, SamplingParams
import argparse
from transformers import AutoTokenizer
import os
from predict_rewriter import load_classifier, predict_label
from sklearn.metrics import classification_report
from tqdm import tqdm
import pandas as pd
import time

def load_model(model_name, temperature):
    
    # Load the default sampling parameters from the model.
    llm = LLM(model=model_name, download_dir='/data/group_data/cx_group/')
    sampling_params = llm.get_default_sampling_params()
    sampling_params.max_tokens = 1024
    sampling_params.temperature =temperature
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return llm, sampling_params, tokenizer

def create_few_shot_prompt(n_shot):
    base_prompt = """Rewrite the following text to make it more engaging and evocative.
    In your rewrite, subtly incorporate an implicit advertisement strategy for {item} within the {type} sector,
    highlighting the following product qualities {qualities}.
    Ensure the ad blends naturally into the text without being overly promotional.
    Return only the revised text with the embedded ad.

    Original Text: {response}
    """

    file_path = "preprocess_data/few_shot_examples_v2.jsonl"
    row = read_jsonl(file_path)
    df = pd.DataFrame(row)

    message = []
    for _, row in df.iterrows():
        item_data = row['item']
        if isinstance(item_data, str):
            item_data = json.loads(item_data)
            
        prompt = base_prompt.format(
            item=item_data['item'],
            type=item_data['type'],
            qualities=item_data['qualities'],
            response=row['response']
        )

        message.append({"role": "user", "content": prompt})
        message.append({"role": "assistant", "content": row['best_answer']})

        n_shot -= 1
        if n_shot == 0:
            break

    return message

    

def create_prompt(row, tokenizer, n_shot=0):
    user_content = f"""Rewrite the following text to make it more engaging and evocative. 
    In your rewrite, subtly incorporate an implict advertisement strategy for '{row['advertisement']}'within the {row['meta_topic']} sector, 
    highlighting the following product qualities {row['qualities']}. 
    Ensure the ad blends naturally into the text without being overly promotional. 
    Return only the revised text with the embedded ad.

    Original Text: '{row['text']}'
    """
    if n_shot > 0:
        message = create_few_shot_prompt(n_shot)
        messages = message + [{"role": "user", "content": user_content}]
    else:
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


def save_generations(generated_texts, rows, output_file):
    """Save generated texts with their prompts to a file"""
    outputs = []
    for i, (text, row) in enumerate(zip(generated_texts, rows)):
        outputs.append({
            "id": i,
            "advertisement": row.get("advertisement", ""),
            "meta_topic": row.get("meta_topic", ""),
            "qualities": row.get("qualities", ""),
            "original_text": row.get("text", ""),
            "generated_text": text
        })
    
    with open(output_file, 'w') as f:
        for output in outputs:
            json.dump(output, f)
            f.write('\n')
    
    print(f"Generated texts saved to {output_file}")
    return outputs
    

def evaluate_with_classifiers(generated_texts):
    """Evaluate texts with multiple classifier versions"""
    results = {}
    
    # Ground truth labels (assuming all should be classified as ads)
    true_labels = [1] * len(generated_texts)
    
    for version in ["0.0", "0.1", "0.2"]:
        print(f"Loading classifier v{version}...")
        clf_model_dir = f"jmvcoelho/ad-classifier-v{version}"
        clf, clf_tokenizer = load_classifier(clf_model_dir)
        
        predictions = []
        for text in tqdm(generated_texts, desc=f"Predicting with classifier v{version}"):
            predictions.append(predict_label(text, clf_tokenizer, clf))
        
        # Calculate metrics
        report = classification_report(true_labels, predictions, output_dict=True, zero_division=0)
        results[version] = {
            "classifier": clf_model_dir,
            "predictions": predictions,
            "report": report,
            "accuracy": report['accuracy'],
            "f1-score": report['weighted avg']['f1-score'],
            "precision": report['weighted avg']['precision'],
            "recall": report['weighted avg']['recall']
        }
    
    return results

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
        default="./data.jsonl",
        help='input data file path')
    parser.add_argument(
        '--n-shot', 
        type=int, 
        default=0,
        help='number of few-shot examples to use')
    parser.add_argument(
        '--temperature', 
        type=float, 
        default=0.5,
        help='temperature for sampling'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=".",
        help='directory to save outputs'
    )
    
    args = parser.parse_args()
    
    file_path = args.file_path
    model_path = args.model_path
    clf_model_dir = args.clf_model_dir
    n_shot = args.n_shot
    temperature = args.temperature
    

    rows = read_jsonl(file_path)

    print("Loading model and tokenizer...")
    llm, sampling_params, tokenizer = load_model(model_path, temperature)

    prompts = []
    for row in rows:
        prompts.append(create_prompt(row, tokenizer, n_shot=n_shot))

    print(f"Generating text with {len(prompts)} prompts...")
    generated_text = generate_text(prompts, llm, sampling_params)

    timestamp = int(time.time())
    generations_file = os.path.join(args.output_dir, f"generations_{timestamp}.jsonl")
    saved_data = save_generations(generated_text, rows, generations_file)


    evaluation_results = evaluate_with_classifiers(generated_text)
    
    # Save evaluation results
    result = {
        "model": model_path,
        "n_shot": n_shot,
        "temperature": temperature,
        "classifier_results": evaluation_results,
        "timestamp": timestamp
    }
    
    results_file = os.path.join(args.output_dir, f"evaluation_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(result, f, indent=2)
    

    # print(f"Loading classifier {clf_model_dir}...")
    # clf, clf_tokenizer = load_classifier(clf_model_dir)

    # predictions = []
    # for text in tqdm(generated_text, desc="Predicting labels"):
    #     predictions.append(predict_label(text, clf_tokenizer, clf))

    # labels = [1] * len(predictions)
    
    # report = classification_report(labels, predictions, output_dict=True, zero_division=0) 
    # result = {
    #     "model": model_path,
    #     "n_shot": n_shot,
    #     "clf_model": clf_model_dir,
    #     "report": report,
    #     "accuracy": report['accuracy'],
    #     "f1-score": report['weighted avg']['f1-score'],
    # }
    # with open('prediction_results.jsonl', 'a') as f:
    #     json.dump(result, f)
    #     f.write("\n")