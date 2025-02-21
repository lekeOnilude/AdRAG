from vllm import LLM, SamplingParams
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from typing import List, Optional
from tqdm import tqdm
import json


class QA_TopicEvaluator:
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        """Initialize the QA model using vLLM."""
        self.model = LLM(model_name)
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)

    def generate_answer(self, query: str, context=None) -> str:
        """Generate an answer to the query using the context."""
        if not context:
            context = "No context provided"
        prompt = f"{query}"
        results = self.model.generate([prompt], self.sampling_params)
        return results[0].outputs[0].text.strip()

    def evaluate(self, query: str, reference_answer: str, context: Optional[List[str]] = None):
        """Evaluate the generated answer using ROUGE and BLEU scores."""
        generated_answer = self.generate_answer(query, context)

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(reference_answer, generated_answer)

        bleu_score = sentence_bleu([reference_answer.split()], generated_answer.split())

        return {
            "query": query,
            "generated_answer": generated_answer,
            "rouge_score": rouge_scores["rougeL"].fmeasure,
            "bleu_score": bleu_score
        }

qa_evaluator = QA_TopicEvaluator()
filepath = "dev_v2.1.json"  # Replace with the correct local path
with open(filepath, "r") as file:
    data = json.load(file)

stats = {
    "count": 0,
    "totalBleu": 0, "minBleu": 1000, "maxBleu": -1000,
    "totalRouge": 0, "minRouge": 1000, "maxRouge": -1000
}

for key in data['wellFormedAnswers']:
    if data['wellFormedAnswers'][key] != '[]':
        if stats["count"] % 100 == 0:
            print(stats)
        query = data["query"][key]
        answer = data['wellFormedAnswers'][key][0]
        results = qa_evaluator.evaluate(query, answer)
        stats["count"] += 1
        stats["totalBleu"] += results["bleu_score"]
        stats["maxBleu"] = max(stats["maxBleu"], results["bleu_score"])
        stats["minBleu"] = min(stats["minBleu"], results["bleu_score"])
        stats["totalRouge"] += results["rouge_score"]
        stats["maxRouge"] = max(stats["maxRouge"], results["rouge_score"])
        stats["minRouge"] = min(stats["minRouge"], results["rouge_score"])

stats['avgBlue'] = stats["totalBleu"] / stats["count"]
stats['avgRouge'] = stats["totalRouge"] / stats["count"]
print(stats)
