
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

# Load the LLM model and tokenizer
model_name = "Qwen/Qwen2.5-7B-Instruct"  # Or another suitable model
# Load the default sampling parameters from the model.
llm = LLM(model=model_name, download_dir='/data/user_data/gonilude')
sampling_params = llm.get_default_sampling_params()
sampling_params.max_tokens = 1024


rows = {}

with open('preprocess_data/valid_with_qualities.jsonl', 'r') as f:
    for line in f:
        row = json.loads(line)
        id_num = row['id']
        # row.pop('id', None)  # Remove the 'id' key from the row
        rows[id_num] = row  # Directly store the row with id as key


def generate_prompt_for_ads_only(row):
    prompt = f""" Write an advertisement for a product using the following information:
    Advertisement Context:
            * Product/Service Category (Meta Topic): {row['meta_topic']}
            * Specific Advertisement Theme: {row['advertisement']}
            * Key Selling Points/Qualities to Emphasize: {row['qualities']}
    The advertisement should be only a sentence long and return only the advertisement.
    """
    
    prompt_info = "<|im_start|>system" \
        "content: You are Qwen, created by Alibaba Cloud. You are a helpful assistant. \n" \
        "<|im_start|>user \n" \
        f"{prompt} \n" \
        "<|im_start|>assistant"
    return prompt_info



def generate_prompt(row):
    # prompt = "Take the following response to a search query and include a subtle advertisement " \
    #     f"for the {row['meta_topic']} {row['advertisement']}. Please follow these instructions: \n" \
    #     f"1. Advertise the qualities: {row['qualities']}. " \
    #     f"You may use different words with relevance to the query as long as the meaning is the same\n" \
    #     "2. Only change one or two consecutive sentences. Do not change the rest of the text.\n" \
    #     "3. Avoid words such as 'like'\n" \
    #     "4. Return only the response with the included ad.\n\n" \
    #     f"Query: {row['query']}\n" \
    #     f"Response:\n\n{row['text']}"
    
    prompt = f"""
            Task: You are tasked with subtly integrating an advertisement into a search query response. The goal is to make the advertisement feel natural and helpful within the context of the response, not disruptive or overtly promotional.

            Advertisement Context:
            * Product/Service Category (Meta Topic): {row['meta_topic']}
            * Specific Advertisement Theme: {row['advertisement']}
            * Key Selling Points/Qualities to Emphasize: {row['qualities']}

            Instructions for Integration:

            1.  Contextual Relevance is Key:  The advertisement must feel relevant to the search query and the existing response text.  Think about how the advertised product/service genuinely relates to the user's needs implied by the query.

            2.  Subtle and Natural Language:  Use language that blends seamlessly with the original response. Avoid overly promotional or sales-oriented phrasing. Focus on informative and helpful language.

            3.  Focus on Benefits, Not Just Features: Instead of directly listing qualities, rephrase them as benefits the user might gain in relation to the search query and response context.

            4.  Strategic Placement: Choose the most natural and impactful location(s) within the response to subtly introduce the advertisement. This might involve:
                * Briefly modifying an existing sentence to subtly incorporate the advertisement.
                * Adding a short, relevant phrase or clause to an existing sentence.
                * In rare cases, adding a very short, contextual sentence (only if it feels truly natural).

            5.  Maintain Original Meaning:  Ensure the core meaning and factual accuracy of the original response remain unchanged. The advertisement should enhance, not distort, the original information.

            6.  Review for Subtlety: Before returning the response, critically evaluate if the advertisement feels genuinely subtle and integrated. If it feels forced or obvious, refine your approach.

            Output: Return only the modified response with the subtly integrated advertisement.

            ---

            Search Query: {row['query']}
            Original Response:

            {row['text']}

            Modified Response:
        """
    
    prompt_info = "<|im_start|>system" \
        "content: You are Qwen, created by Alibaba Cloud. You are a helpful assistant. \n" \
        "<|im_start|>user \n" \
        f"{prompt} \n" \
        "<|im_start|>assistant"
    return prompt_info

prompts = []
ids = []
original_responses = []
for id, row in rows.items():
    prompts.append(generate_prompt_for_ads_only(row))
    ids.append(id)
    original_responses.append(row['text'])


outputs = llm.generate(prompts, sampling_params)

results = []
for id, text,output in zip(ids, original_responses, outputs):
    result = {}
    result["id"] = id
    prompt = output.prompt
    generated_text = output.outputs[0].text
    result['generated_text'] = f"{text} {generated_text}"
    results.append(result)

with open("rewritten_responses_append_only.json", "w") as outfile:
    for result in results:
        json.dump(result, outfile)
        outfile.write("\n")

    


# results = []
# for id, row in rows.items():
#     result = {}
#     rewritten_text = generate_response(row)
#     result[id] = rewritten_text
#     results.append(result)


# # Save the rewritten responses (optional):
# with open("rewritten_responses.json", "w") as outfile:
#     for result in results:
#         json.dump(result, outfile)
#         outfile.write("\n")
