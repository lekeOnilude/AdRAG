
import json
from vllm import LLM, SamplingParams
import argparse

# Load the LLM model and tokenizer
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct" #"Qwen/Qwen2.5-7B-Instruct"  # Or another suitable model

def load_model(model_name):
    
    # Load the default sampling parameters from the model.
    llm = LLM(model=model_name, download_dir='/data/user_data/gonilude')
    sampling_params = llm.get_default_sampling_params()
    sampling_params.max_tokens = 1024
    return llm, sampling_params



def generate_prompt_for_ads_only(row):
    prompt = f""" Write an advertisement for a product using the following information:
    Advertisement Context:
            * Product/Service Category (Meta Topic): {row['meta_topic']}
            * Specific Advertisement Theme: {row['advertisement']}
            * Key Selling Points/Qualities to Emphasize: {row['qualities']}
    The advertisement should be only a sentence long and return only the advertisement.
    """
    
    prompt_info = "<|im_start|>system" \
        "content:You are a helpful assistant. \n" \
        "<|im_start|>user \n" \
        f"{prompt} \n" \
        "<|im_start|>assistant"
    return prompt_info



def generate_prompt(row, few_shot=None):
    prompt = f"""Take the following response to a search query and include a subtle advertisement 
        for the {row['meta_topic']} {row['advertisement']}. Please follow these instructions:
        1. Advertise the qualities: {row['qualities']}. " You may use different words with relevance to the query as long as the meaning is the same
        2. Only change one or two consecutive sentences. Do not change the rest of the text.
        3. Avoid words such as 'like'
        4. Return only the response with the included ad.

        Query: {row['query']}
        Response:
        
        {row['text']}
    """

    if few_shot is not None:
        prompt = few_shot + prompt
    
    # prompt = f"""
    #         Task: You are tasked with subtly integrating an advertisement into a search query response. The goal is to make the advertisement feel natural and helpful within the context of the response, not disruptive or overtly promotional.

    #         Advertisement Context:
    #         * Product/Service Category (Meta Topic): {row['meta_topic']}
    #         * Specific Advertisement Theme: {row['advertisement']}
    #         * Key Selling Points/Qualities to Emphasize: {row['qualities']}

    #         Instructions for Integration:

    #         1.  Contextual Relevance is Key:  The advertisement must feel relevant to the search query and the existing response text.  Think about how the advertised product/service genuinely relates to the user's needs implied by the query.

    #         2.  Subtle and Natural Language:  Use language that blends seamlessly with the original response. Avoid overly promotional or sales-oriented phrasing. Focus on informative and helpful language.

    #         3.  Focus on Benefits, Not Just Features: Instead of directly listing qualities, rephrase them as benefits the user might gain in relation to the search query and response context.

    #         4.  Strategic Placement: Choose the most natural and impactful location(s) within the response to subtly introduce the advertisement. This might involve:
    #             * Briefly modifying an existing sentence to subtly incorporate the advertisement.
    #             * Adding a short, relevant phrase or clause to an existing sentence.
    #             * In rare cases, adding a very short, contextual sentence (only if it feels truly natural).

    #         5.  Maintain Original Meaning:  Ensure the core meaning and factual accuracy of the original response remain unchanged. The advertisement should enhance, not distort, the original information.

    #         6.  Review for Subtlety: Before returning the response, critically evaluate if the advertisement feels genuinely subtle and integrated. If it feels forced or obvious, refine your approach.

    #         Output: Return only the modified response with the subtly integrated advertisement.

    #         ---

    #         Search Query: {row['query']}
    #         Original Response:

    #         {row['text']}

    #         Modified Response:
    #     """
    
    # prompt_info = "<|im_start|>system" \
    #     "content: You are a helpful assistant. \n" \
    #     "<|im_start|>user \n" \
    #     f"{prompt} \n" \
    #     "<|im_start|>assistant"
    
    prompt_info = "<bos><start_of_turn>user" \
            f"{prompt} \n" \
            "<start_of_turn>model"
    return prompt_info

# prompts = []
# ids = []
# original_responses = []
# for id, row in rows.items():
#     prompts.append(generate_prompt_for_ads_only(row))
#     ids.append(id)
#     original_responses.append(row['text'])


# outputs = llm.generate(prompts, sampling_params)

# results = []
# for id, text,output in zip(ids, original_responses, outputs):
#     result = {}
#     result["id"] = id
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     result['generated_text'] = f"{text} {generated_text}"
#     results.append(result)

# with open("rewritten_responses_append_only.json", "w") as outfile:
#     for result in results:
#         json.dump(result, outfile)
#         outfile.write("\n")


def generate_fewshot(fewshot_path, n_shots=2):

    if n_shots == 0:
        return None

    fewshot_prompts = ""
    with open(fewshot_path, 'r') as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            prompt = f"""Take the following response to a search query and include a subtle advertisement 
                for the {row['meta_topic']} {row['advertisement']}. Please follow these instructions:
                1. Advertise the qualities: {row['qualities']}. " You may use different words with relevance to the query as long as the meaning is the same
                2. Only change one or two consecutive sentences. Do not change the rest of the text.
                3. Avoid words such as 'like'
                4. Return only the response with the included ad.

                Query: {row['query']}
                Response:
                {row['text']}

                Result:
                {row['generated_text']}
            """ 

            if i == 0:
                fewshot_prompts = prompt
            else:
                fewshot_prompts = f"{fewshot_prompts} \n\n {prompt}"

            if i+1 >= n_shots:
                break
    
    fewshot_prompts = fewshot_prompts + "\n\n Your turn: \n" 

    return fewshot_prompts



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--n_shots", type=int, default=0)
    args = parser.parse_args()

    model_name = args.model_name
    n_shots= args.n_shots

    llm, sampling_params = load_model(model_name)

    rows = {}
    with open('preprocess_data/valid_with_qualities.jsonl', 'r') as f:
        for line in f:
            row = json.loads(line)
            id_num = row['id']
            rows[id_num] = row 

    prompts = []
    ids = []
    few_shot = generate_fewshot('preprocess_data/few_shot_examples.jsonl', n_shots=n_shots)
    for id, row in rows.items():
        prompts.append(generate_prompt(row, few_shot=few_shot))
        ids.append(id)

    outputs = llm.generate(prompts, sampling_params)

    results = []
    for id, output in zip(ids, outputs):
        result = {}
        result["id"] = id
        prompt = output.prompt
        generated_text = output.outputs[0].text
        result['generated_text'] = generated_text
        results.append(result)

    # Save the rewritten responses (optional):
    with open(f"rewritten_responses_{model_name.split('/')[-1]}_fewshot_{n_shots}.json", "w") as outfile:
        for result in results:
            json.dump(result, outfile)
            outfile.write("\n")
