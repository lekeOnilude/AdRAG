import torch
from transformers import AutoTokenizer
import random
from tevatron.retriever.modeling import DenseModel, DenseModelWithNegativeCache

from tevatron.retriever.arguments import (
    ModelArguments,
    DataArguments,
    TevatronTrainingArguments as TrainingArguments,
)

from torch.nn.utils import clip_grad_norm_

import copy

import torch.nn.functional as F
from datasets import load_dataset

import os

import logging

import json


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

import pickle
from tqdm import tqdm

from transformers import (
    HfArgumentParser,
)

import sys


class MATESQueryAttribution:
    def __init__(
        self,
        valid_dataset_path,
        use_negative_cache,
        n_valid_queries,
        n_valid_negs,
        model_args,
        training_args,
        data_args,
    ):

        self.data_args = data_args

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        self.tokenizer.padding_side = "right"

        self.parse_valid_dataset(
            num_queries=n_valid_queries,
            negs_per_query=n_valid_negs,
            jsonl_path=valid_dataset_path,
        )

        data_files = [f"en_{str(i).zfill(2)}.jsonl" for i in range(24)]
        self.dataset = load_dataset(
            "XBKYS/minicpm-embedding-data",
            data_files=data_files,
            split="train",
            cache_dir=data_args.dataset_cache_dir,
        )

        print(f"Sharding in {data_args.dataset_number_of_shards} shards.")
        print(f"Processing shard {data_args.dataset_shard_index}.")
        self.dataset = self.dataset.shard(
            num_shards=data_args.dataset_number_of_shards,
            index=data_args.dataset_shard_index,
        )

        if not use_negative_cache:
            self.model = DenseModel.build(
                model_args,
                training_args,
                cache_dir=model_args.cache_dir,
            )

            self.model.to("cuda")
            self.model_copy = copy.deepcopy(self.model)

        else:
            logger.info("Using Negative Cache")
            self.model = DenseModelWithNegativeCache.build(
                model_args,
                training_args,
                cache_dir=model_args.cache_dir,
            )

            self.model.to("cuda")
            self.model_copy = copy.deepcopy(self.model)
            self.model_copy.negative_cache = None
            self.build_negative_cache()

    def set_seed(self, seed):
        random.seed(seed)

    def build_negative_cache(self, batch_size=300, cache_size=2874):
        self.model.eval()

        docs = []
        while len(docs) < cache_size:
            random_index = random.randint(0, len(self.dataset) - 1)
            random_negative_set = self.dataset[random_index]["neg"]
            random_negative_index = random.randint(
                1, len(random_negative_set) - 1
            )  # 0th is a prompt
            docs.append(random_negative_set[random_negative_index])

        all_reps = []
        with torch.no_grad():
            for i in range(0, len(docs), batch_size):
                batch_docs = docs[i : i + batch_size]

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    _, tokenized_docs = self.tokenize(q=None, d=batch_docs)
                    tokenized_docs = {
                        k: v.to("cuda") for k, v in tokenized_docs.items()
                    }
                    batch_reps = self.model(passage=tokenized_docs).p_reps.detach()
                    all_reps.append(batch_reps)

        cache_reps = torch.cat(all_reps, dim=0)
        assert (
            cache_reps.size(0) == cache_size
        ), f"Cache size mismatch: {cache_reps.size(0)} vs {cache_size}"
        self.model.init_cache(cache_reps)
        self.model.train()

    def parse_valid_dataset(self, num_queries, negs_per_query, jsonl_path):
        valid_samples = []
        with open(jsonl_path, "r") as file:
            for line in file:
                json_data = json.loads(line.strip())
                valid_samples.append(json_data)

        valid_samples = [
            valid_samples[i : i + num_queries]
            for i in range(0, len(valid_samples), num_queries)
        ]

        self.all_valid_groups = []
        for valid_group in valid_samples:
            v_queries = []
            v_documents = []
            for valid_instance in valid_group:
                v_queries.append(self.tokenizer.decode(valid_instance["query"]))
                v_documents.append(
                    self.tokenizer.decode(valid_instance["positives"][0])
                )
                v_documents += self.tokenizer.batch_decode(
                    valid_instance["negatives"][:negs_per_query]
                )

            qv, dv = self.tokenize(v_queries, v_documents)

            qv = {k: v.to("cuda") for k, v in qv.items()}
            dv = {k: v.to("cuda") for k, v in dv.items()}
            self.all_valid_groups.append((qv, dv))

        self.all_valid_groups_full_negs = []
        for valid_group in valid_samples:
            v_queries = []
            v_documents = []
            for valid_instance in valid_group:
                v_queries.append(self.tokenizer.decode(valid_instance["query"]))
                v_documents.append(
                    self.tokenizer.decode(valid_instance["positives"][0])
                )
                v_documents += self.tokenizer.batch_decode(
                    valid_instance["negatives"][:6]
                )

            qv, dv = self.tokenize(v_queries, v_documents)

            qv = {k: v.to("cuda") for k, v in qv.items()}
            dv = {k: v.to("cuda") for k, v in dv.items()}
            self.all_valid_groups_full_negs.append((qv, dv))

        print(len(self.all_valid_groups))
        print(len(self.all_valid_groups_full_negs))

    def tokenize(self, q, d):

        if q is not None:
            q_collated = self.tokenizer(
                q,
                padding=False,
                truncation=True,
                max_length=(
                    self.data_args.query_max_len - 1
                    if self.data_args.append_eos_token
                    else self.data_args.query_max_len
                ),
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=True,
            )

            if self.data_args.append_eos_token:
                q_collated["input_ids"] = [
                    q + [self.tokenizer.eos_token_id] for q in q_collated["input_ids"]
                ]

            q_collated = self.tokenizer.pad(
                q_collated,
                padding=True,
                pad_to_multiple_of=self.data_args.pad_to_multiple_of,
                return_attention_mask=True,
                return_tensors="pt",
            )
        else:
            q_collated = None

        if d is not None:
            d_collated = self.tokenizer(
                d,
                padding=False,
                truncation=True,
                max_length=(
                    self.data_args.passage_max_len - 1
                    if self.data_args.append_eos_token
                    else self.data_args.passage_max_len
                ),
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=True,
            )

            d_collated["input_ids"] = [
                d + [self.tokenizer.eos_token_id] for d in d_collated["input_ids"]
            ]

            d_collated = self.tokenizer.pad(
                d_collated,
                padding=True,
                pad_to_multiple_of=self.data_args.pad_to_multiple_of,
                return_attention_mask=True,
                return_tensors="pt",
            )
        else:
            d_collated = None

        return q_collated, d_collated

    def get_subsetvalid_loss(self, q, d, qv, dv):

        self.model.train()

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):  # HACK hardcoded to bf16
            # TRAIN
            q = {k: v.to("cuda") for k, v in q.items()}
            d = {k: v.to("cuda") for k, v in d.items()}

            loss = self.model(q, d).loss

            loss.backward(gradient=torch.ones_like(loss))

        clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        optimizer.step()
        optimizer.zero_grad()

        self.model.eval()

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss = self.model(qv, dv, use_cache=False).loss

        final = loss.item()

        self.model.load_state_dict(self.model_copy.state_dict())
        self.model.train()
        return final

    def get_subset_and_full_valid_loss(self, q, d, qv, dv, D):

        self.model.train()

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):  # HACK hardcoded to bf16
            # TRAIN
            q = {k: v.to("cuda") for k, v in q.items()}
            d = {k: v.to("cuda") for k, v in d.items()}

            loss = self.model(q, d).loss

            loss.backward(gradient=torch.ones_like(loss))

        clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        optimizer.step()
        optimizer.zero_grad()

        self.model.eval()

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss = self.model(qv, dv, use_cache=False).loss

        subset_loss = loss.item()

        all_losses = []
        for qv, dv in D:
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    loss = self.model(qv, dv, use_cache=False).loss
                    all_losses.append(loss.item())

        total_loss = sum(all_losses) / len(all_losses)

        self.model.load_state_dict(self.model_copy.state_dict())
        self.model.train()
        return subset_loss, total_loss, all_losses

    def get_mates_score(self, example) -> list[str]:

        query_text = example["query"][1]
        positive_text = example["pos"][1]
        negative_texts = example["neg"][1:]

        queries = [query_text]
        documents = [positive_text] + negative_texts

        q, d = self.tokenize(queries, documents)

        # valid_instances_group = random.randint(0, len(self.all_valid_groups) - 1)
        valid_instances_group = 14
        qv, dv = self.all_valid_groups[valid_instances_group]

        valid_loss = self.get_subsetvalid_loss(q, d, qv, dv)

        return valid_loss, valid_instances_group

    def run(self, outpath):
        logger.info(f"Sampling started. Saving to: {outpath}")

        completed_examples = 0

        if os.path.exists(outpath):
            with open(outpath, "r") as h1:
                completed_examples = sum(1 for _ in h1)
            logger.info(
                f"File '{outpath}' already exists with {completed_examples} entries. Resuming from there."
            )
        else:
            logger.info(
                f"File '{outpath}' does not exist. Starting sampling from the beginning."
            )

        dataset_to_process = self.dataset.select(
            range(completed_examples, len(self.dataset))
        )

        with open(outpath, "a") as h1:
            for i, example in enumerate(tqdm(dataset_to_process)):

                query = (
                    example["query"][1]
                    .replace("\t", "")
                    .replace("\n", "")
                    .replace("\r", "")
                )

                valid_loss, valid_instances_group = self.get_mates_score(example)

                h1.write(f"{query}\t{valid_loss}\t{valid_instances_group}\n")

                if i % 10 == 0:
                    h1.flush()

    def get_valid_corr(self, outpath):

        sampled_dataset = self.dataset.shuffle(seed=42).select(range(75))
        with open(outpath, "w") as h1:
            for i, example in enumerate(tqdm(sampled_dataset), 1):

                query_text = example["query"][1]
                positive_text = example["pos"][1]
                negative_texts = example["neg"][1:]

                queries = [query_text]
                documents = [positive_text] + negative_texts

                q, d = self.tokenize(queries, documents)

                valid_instances_group = 14
                qv, dv = self.all_valid_groups[valid_instances_group]

                subset_loss, total_loss, all_full_batch_losses = (
                    self.get_subset_and_full_valid_loss(
                        q, d, qv, dv, self.all_valid_groups_full_negs
                    )
                )

                query = (
                    example["query"][1]
                    .replace("\t", "")
                    .replace("\n", "")
                    .replace("\r", "")
                )

                h1.write(
                    f"{query}\t{subset_loss}\t{total_loss}\t{','.join([str(x) for x in all_full_batch_losses])}\n"
                )


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 3 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    N_VALID_Q = 100
    N_VALID_N = 2

    attribution_method = MATESQueryAttribution(
        valid_dataset_path="/data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-avg-pool-mntp-finetune-ep1/pretokenized/val_shuf_subset.jsonl",
        use_negative_cache=True,
        n_valid_queries=N_VALID_Q,
        n_valid_negs=N_VALID_N,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )

    attribution_method.set_seed(17121998)
    # attribution_method.get_valid_corr(
    #     outpath=f"{training_args.output_dir}_{data_args.dataset_shard_index}"
    # )
    attribution_method.run(
        outpath=f"{training_args.output_dir}_{data_args.dataset_shard_index}"
    )


if __name__ == "__main__":
    main()
