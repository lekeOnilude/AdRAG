import torch
from transformers import AutoTokenizer
import random
from tevatron.retriever.modeling import DenseModel, DenseModelWithNegativeCache

from tevatron.retriever.arguments import (
    ModelArguments,
    DataArguments,
    TevatronTrainingArguments as TrainingArguments,
)

import torch.nn.functional as F
from datasets import load_dataset

import os

import logging

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


class LESSQueryAttribution:
    def __init__(
        self,
        valid_gradient_path,
        optim_state_path,
        use_negative_cache,
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

        self.validation_gradient = self.get_validation_gradient(
            valid_gradient_path
        ).cuda()
        optimizer_states = torch.load(optim_state_path, map_location="cpu")
        self.avg = optimizer_states["optimizer_state_dict"]["optimizer_state_dict"][
            "state"
        ][0]["exp_avg"].cuda()
        self.avg_sq = optimizer_states["optimizer_state_dict"]["optimizer_state_dict"][
            "state"
        ][0]["exp_avg_sq"].cuda()

        del optimizer_states

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

        else:
            logger.info("Using Negative Cache")
            self.model = DenseModelWithNegativeCache.build(
                model_args,
                training_args,
                cache_dir=model_args.cache_dir,
            )

            self.model.to("cuda")

            self.build_negative_cache()

    def set_seed(self, seed):
        random.seed(seed)

    def build_negative_cache(
        self, batch_size=300, cache_size=2874
    ):  # (8gpu * 60 batch size * 6 docs/query) - 6
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

    def get_validation_gradient(self, valid_gradient_path):
        with open(valid_gradient_path, "rb") as h:
            grad = pickle.load(h).float()

        print(
            f"Loaded valid gradient. Shape: {grad.shape}. Norm: {torch.norm(grad, p=2).item()}"
        )
        return grad

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

    def get_grad_dotp(self, q, d, do_adam=False):

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):  # HACK hardcoded to bf16

            q = {k: v.view(1, -1).to("cuda") for k, v in q.items()}
            d = {k: v.to("cuda") for k, v in d.items()}

            loss = self.model(q, d).loss

            loss.backward()

            gradients = []
            for param in self.model.parameters():
                if param.grad is not None:
                    gradients.append(param.grad.detach().view(-1))

            gradient_vector = torch.cat(gradients)

            if do_adam:
                beta1 = 0.9
                beta2 = 0.999
                eps = 1e-08

                updated_avg = beta1 * self.avg + (1 - beta1) * gradient_vector
                updated_avg_sq = beta2 * self.avg_sq + (1 - beta2) * gradient_vector**2
                gradient_vector = updated_avg / torch.sqrt(updated_avg_sq + eps)

            grad_norm = torch.norm(gradient_vector, p=2).item()
            gradient_vector = F.normalize(gradient_vector, p=2, dim=0)

            dot_product = torch.dot(gradient_vector, self.validation_gradient)

            self.model.zero_grad()

        return dot_product, grad_norm

    def get_less_score(self, example) -> list[str]:

        query_text = example["query"][1]
        positive_text = example["pos"][1]
        negative_texts = example["neg"][1:]

        queries = [query_text]
        documents = [positive_text] + negative_texts

        q, d = self.tokenize(queries, documents)

        dot_prod, grad_norm = self.get_grad_dotp(q, d)

        return dot_prod.item(), grad_norm

    def run(self, outpath):
        logger.info(f"Sampling started. Saving to: {outpath}")

        with open(outpath, "w") as h1:
            for example in tqdm(self.dataset):

                query = (
                    example["query"][1]
                    .replace("\t", "")
                    .replace("\n", "")
                    .replace("\r", "")
                )

                dot_product, grad_norm = self.get_less_score(example)
                h1.write(f"{query}\t{dot_product}\t{grad_norm}\n")


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    attribution_method = LESSQueryAttribution(
        valid_gradient_path="/data/user_data/jmcoelho/models/Qwen2.5-0.5B-bidirectional-attn-wavg-pool-mntp-minicpmembed-random-20k-1gpu-valid-grads/normalized_final_valid_grad.pkl",
        optim_state_path="/data/user_data/jmcoelho/models/Qwen2.5-0.5B-bidirectional-attn-wavg-pool-mntp-minicpmembed-random-20k-1gpu/checkpoint-167/global_step167/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt",
        use_negative_cache=True,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )

    attribution_method.set_seed(17121998)
    attribution_method.run(
        outpath=f"{training_args.output_dir}_{data_args.dataset_shard_index}"
    )


if __name__ == "__main__":
    main()
