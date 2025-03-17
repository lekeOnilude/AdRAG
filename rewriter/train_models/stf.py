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

PATH_TO_DATA = "/data/user_data/jmcoelho/datasets/llama_generator/gpt_3.5_turbo_sft_dataset.tsv"

def create_hf_dataset_from_tsv(tsv_file):
    df = pd.read_csv(tsv_file, sep="\t")
    hf_dataset = Dataset.from_pandas(df)
    return hf_dataset

def load_model_and_tokenizer(model_path: str = "/data/models/huggingface/meta-llama/Meta-Llama-3-8B-Instruct"):
    """Load model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )

    model.print_trainable_parameters()

    return model, tokenizer

if __name__ == "__main__":
    OUTPATH = "/data/user_data/jmcoelho/models/query_generators/sft_gpt_pairs/"
    model, tokenizer = load_model_and_tokenizer()

    dataset = create_hf_dataset_from_tsv(PATH_TO_DATA)
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
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
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