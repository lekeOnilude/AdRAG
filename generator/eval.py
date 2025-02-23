import json
import glob
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tqdm import tqdm

QUERY_SET = "marcov2"
RETRIEVAL_MODEL = "Qwen2.5-0.5B-bidirectional-attn-mntp-marco-passage-hard-negatives-matrioshka-reduction-2"
RUN_NAME = "Qwen2.5-1.5B-Instruct-10-passage-RAG"


generated_files_pattern = f"/home/jmcoelho/11797_Project/generator/output/{QUERY_SET}/{RETRIEVAL_MODEL}/{RUN_NAME}_*.jsonl"
print(generated_files_pattern)
generated_results = []
for filepath in glob.glob(generated_files_pattern):
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                generated_results.append(json.loads(line))

print(f"Loaded {len(generated_results)} generated results.")


golden_file = f"/home/jmcoelho/11797_Project/retrieval/output/{QUERY_SET}/{RETRIEVAL_MODEL}.jsonl"  # change this to the actual path
golden_dict = {}
with open(golden_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            query_id = data["query_id"]
            golden_dict[query_id] = data["answers"]

print(f"Loaded golden answers for {len(golden_dict)} queries.")

bleu1_scores = []
rougeL_scores = []
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

for item in tqdm(generated_results):
    query_id = item["query_id"]
    generated_text = item["response"]

    # Retrieve the corresponding golden answers (a list of reference strings).
    references = golden_dict.get(query_id, [])
    if not references:
        print(f"Warning: No golden answers found for query_id {query_id}. Skipping.")
        continue

    # Tokenize the generated text and each reference for BLEU-1 computation.
    gen_tokens = word_tokenize(generated_text)
    ref_tokens = [word_tokenize(ref) for ref in references]

    # Compute BLEU-1 (unigram precision)
    try:
        bleu1 = sentence_bleu(
            ref_tokens,
            gen_tokens,
            weights=(1, 0, 0, 0),
            smoothing_function=SmoothingFunction().method1,
        )
    except ZeroDivisionError:
        bleu1 = 0.0
    bleu1_scores.append(bleu1)

    # Compute ROUGE-L for each reference and average the scores.
    temp_rougeL = [
        scorer.score(generated_text, ref)["rougeL"].fmeasure for ref in references
    ]
    avg_rougeL = sum(temp_rougeL) / len(temp_rougeL) if temp_rougeL else 0.0

    rougeL_scores.append(avg_rougeL)

# Calculate and report the overall average scores.
overall_bleu1 = sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0.0
overall_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0

print("Average BLEU-1 score:", overall_bleu1)
print("Average ROUGE-L score:", overall_rougeL)
