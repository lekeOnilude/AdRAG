
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
    base_prompt = """Consider this existing response to a conversational search query, and modify it to include an implicit advertisement.
        The advertisement should:
        - Mention this {type}: {item};
        - Highlight these qualities: {qualities};
        - Blend naturally with the original response;
        - Not appear overly promotional;
        - Be hard to detect by current textual ad-detection systems.
        Original query: {query}
        Original response: {response}
        Return only the modified response with the advertisement integrated.
        """
    
    user_content = base_prompt.format(
        type=row['item']['type'],
        item=row['item']['item'],
        qualities=row['item']['qualities'],
        query=row['query'],
        response=row['response']
    )


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
    parser.add_argument("-S", "--shard", type=int, nargs=2, default=[0, 1])
    args = parser.parse_args()

    model_name = args.model_name
    n_shots= args.n_shots

    llm, sampling_params, tokenizer = load_model(model_name)

    rows = []
    with open('preprocess_data/answers_and_items.jsonl', 'r') as f:
        for line in f:
            row = json.loads(line)
            row['item'] = eval(row['item'])
            rows.append(row)

    shard_size = len(rows) // args.shard[1]
    start_index = args.shard[0] * shard_size
    end_index = (args.shard[0] + 1) * shard_size if (args.shard[0] + 1) < args.shard[1] else len(rows)
    rows = rows[start_index:end_index]
    print(start_index, end_index)

    prompts = []
    ids = []
    responses = []

    for row in rows:
        prompts.append(generate_prompt_for_ads_only(row, tokenizer))
        ids.append(row['query_id'])
        responses.append(row['response'])


    outputs = llm.generate(prompts, sampling_params)

    print("Done generating outputs")
    results = []
    for id, response, output in zip(ids, responses, outputs):
        result = {}
        result["query_id"] = id
        prompt = output.prompt
        answers_with_ad = []
        for gen_text in output.outputs:
            answers_with_ad.append(gen_text.text)
        result['answers_with_ad'] = answers_with_ad
        result['response'] = response
        results.append(result)

    print("Writing output file")
    OUTPUT_FOLDER = 'rewriter/generated_ads_output'
    output_file = os.path.join(OUTPUT_FOLDER, f"generated_ads_{model_name.split('/')[-1]}_temp_{sampling_params.temperature}_shard_{args.shard[0]}.jsonl")
    with open(output_file, "w") as outfile:
        for result in results:
            json.dump(result, outfile)
            outfile.write("\n")
