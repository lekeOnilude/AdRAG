# -*- coding: utf-8 -*-
import json
from dataclasses import dataclass, field
from typing import Optional
from trl import SFTTrainer, SFTConfig
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import pandas as pd

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

PATH_TO_DATA = (
    f"/home/jmcoelho/11797_Project/rewriter/output/marcov2.train.set2/answers_with_ads/generated_ads_Qwen2.5-7B-Instruct_temp_1.0_stf.jsonl"
)

base_prompt = """Rewrite the following text to make it more engaging and evocative.
In your rewrite, subtly incorporate an implicit advertisement strategy for {item} within the {type} sector,
highlighting the following product qualities {qualities}.
Ensure the ad blends naturally into the text without being overly promotional.
Return only the revised text with the embedded ad.

Original Text: {response}
"""


def create_hf_dataset_from_jsonl(jsonl_file):
    """Create a HuggingFace dataset from a JSONL file."""
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    
    formatted_data = []
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
        
        formatted_data.append({
            "prompt": prompt,
            "chosen": row['best_answer'],
        })
    
    formatted_df = pd.DataFrame(formatted_data)
    hf_dataset = Dataset.from_pandas(formatted_df)
    return hf_dataset

def load_model_and_tokenizer(
    model_path: str = "Qwen/Qwen2.5-7B-Instruct",
):
    """load model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", use_cache=False
    )

    return model, tokenizer

if __name__ == "__main__":
    OUTPATH = "/data/user_data/jmcoelho/models/ad_writer/dpo_it0/"
    model, tokenizer = load_model_and_tokenizer()

    dataset = create_hf_dataset_from_jsonl(PATH_TO_DATA)
    dataset = dataset.map(
        lambda row: {
            "prompt": [{"role": "user", "content": row["prompt"]}],
            "completion": [{"role": "assistant", "content": row["chosen"]}],
        }
    )

    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    args = SFTConfig(
        output_dir=OUTPATH,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        bf16=True,
        learning_rate=1e-5,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_steps=10000,
        evaluation_strategy="steps",
        eval_steps=100,
        max_length=2048,
        max_prompt_length=1536,
        max_completion_length=512,
        report_to="wandb",
        run_name="sft_test",
        deepspeed="./deepspeed/ds.json",
    )

    trainer = SFTTrainer(
        model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model()