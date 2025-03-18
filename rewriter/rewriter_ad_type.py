
import json
from vllm import LLM, SamplingParams
import argparse
from transformers import AutoTokenizer
import os


# Load the LLM model and tokenizer
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct" #"Qwen/Qwen2.5-7B-Instruct"  # Or another suitable model

def load_model(model_name):
    
    # Load the default sampling parameters from the model.
    llm = LLM(model=model_name, download_dir='/data/group_data/cx_group/')
    sampling_params = llm.get_default_sampling_params()
    sampling_params.max_tokens = 1024
    sampling_params.temperature =1.0
    sampling_params.n = 5
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return llm, sampling_params, tokenizer



def generate_prompt_for_ads_only(row, tokenizer):
    nl = "\n"
    user_content = f"""Rewrite the following text to make it more engaging and evocative. 
    In your rewrite, subtly incorporate an implict advertisement strategy for '{row['advertisement']}'within the {row['meta_topic']} sector, 
    highlighting the following product qualities {row['qualities']}. 
    Ensure the ad blends naturally into the text without being overly promotional. 
    Return only the revised text with the embedded ad.

    Original Text: '{row['text']}'
    """

    # user_content = f"""Take the following response to a search query and include a subtle advertisement 
    #     for the {row['meta_topic']} {row['advertisement']}. Please follow these instructions:
    #     1. Advertise the qualities: {row['qualities']}. " You may use different words with relevance to the query as long as the meaning is the same
    #     2. Only change one or two consecutive sentences. Do not change the rest of the text.
    #     3. Avoid words such as 'like'
    #     4. Return only the response with the included ad.

    #     Response:
        
    #     {row['text']}
    # """


    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. For this interaction, you will be rewriting a text to include an advertisement.",
        },
        {"role": "user", "content": user_content},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt_text




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n_shots", type=int, default=0)
    args = parser.parse_args()

    model_name = args.model_name
    n_shots= args.n_shots

    llm, sampling_params, tokenizer = load_model(model_name)

    rows = []
    with open('preprocess_data/data.jsonl', 'r') as f:
        for line in f:
            row = json.loads(line)
            rows.append(row)


    prompts = []
    ids = []
    meta_topics = []
    few_shot = None #generate_fewshot('preprocess_data/few_shot_examples.jsonl', n_shots=n_shots)
    for row in rows:
        prompts.append(generate_prompt_for_ads_only(row, tokenizer))
        ids.append(row['id'])
        meta_topics.append(row['meta_topic'])

    outputs = llm.generate(prompts, sampling_params)

    results = []
    for id, meta_topic, output in zip(ids, meta_topics, outputs):
        result = {}
        result["id"] = id
        prompt = output.prompt
        generated_text = output.outputs[0].text
        result['generated_text'] = generated_text
        result['meta_topic'] = meta_topic

        results.append(result)

    OUTPUT_FOLDER = 'rewriter/rewritten_response_output'
    output_file = os.path.join(OUTPUT_FOLDER, f"baseline_ads_{model_name.split('/')[-1]}_fewshot_{n_shots}_temp_{sampling_params.temperature}.jsonl")
    with open(output_file, "w") as outfile:
        for result in results:
            json.dump(result, outfile)
            outfile.write("\n")
