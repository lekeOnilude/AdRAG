import logging
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import nn, Tensor

from tevatron.retriever.modeling import DenseModel, EncoderOutput
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from safetensors.torch import load_file as load_safetensors
import os
from tevatron.retriever.arguments import (
    ModelArguments,
    TevatronTrainingArguments as TrainingArguments,
)

logger = logging.getLogger(__name__)


class DenseWithDimReduction(DenseModel):
    def __init__(
        self,
        encoder,
        dim_reduction_factor,
        pooling="cls",
        normalize=False,
        temperature=1.0,
    ):
        # Initialize the base DenseModel.
        super().__init__(
            encoder, pooling=pooling, normalize=normalize, temperature=temperature
        )
        # Set up the linear projection layer.
        logger.info(
            f"Model uses dimensional reduction head. From {self.config.hidden_size} to {self.config.hidden_size // int(dim_reduction_factor)}."
        )
        original_dim = self.config.hidden_size
        reduced_dim = original_dim // int(dim_reduction_factor)
        self.encoder.dim_reduct_linear_projection = nn.Linear(original_dim, reduced_dim)

    def forward(
        self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None
    ):
        # Compute the full embeddings using DenseModel's encode methods.
        q_reps = self.encode_query(query) if query is not None else None
        p_reps = self.encode_passage(passage) if passage is not None else None

        # Compute the lower-dimensional representations.
        q_reps = (
            self.encoder.dim_reduct_linear_projection(q_reps).float()
            if query is not None
            else None
        )
        p_reps = (
            self.encoder.dim_reduct_linear_projection(p_reps).float()
            if passage is not None
            else None
        )

        # If either input is missing, return the available outputs.
        if q_reps is None or p_reps is None:
            return EncoderOutput(q_reps=q_reps, p_reps=p_reps)

        # Training mode: compute losses on both representations.
        if self.training:
            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            # Full embedding similarity and loss.
            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)
            target = torch.arange(
                scores.size(0), device=scores.device, dtype=torch.long
            )
            target = target * (p_reps.size(0) // q_reps.size(0))
            loss = self.compute_loss(scores / self.temperature, target)

            if self.is_ddp:
                loss = loss * self.world_size  # Adjust loss for DDP.
        else:
            # In eval mode, only compute similarity on the full embeddings.
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None

        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    @classmethod
    def build(
        cls,
        model_args: ModelArguments,
        train_args: TrainingArguments,
        **hf_kwargs,
    ):
        base_model = cls.TRANSFORMER_CLS.from_pretrained(
            model_args.model_name_or_path,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            **hf_kwargs,
        )

        print(f"Model class: {base_model.__class__}")

        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if model_args.lora or model_args.lora_name_or_path:
            if train_args.gradient_checkpointing:
                base_model.enable_input_require_grads()
            if model_args.lora_name_or_path:
                lora_config = LoraConfig.from_pretrained(
                    model_args.lora_name_or_path, **hf_kwargs
                )
                lora_model = PeftModel.from_pretrained(
                    base_model, model_args.lora_name_or_path, is_trainable=True
                )
            else:
                lora_config = LoraConfig(
                    base_model_name_or_path=model_args.model_name_or_path,
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules=model_args.lora_target_modules.split(","),
                    inference_mode=False,
                )
                lora_model = get_peft_model(base_model, lora_config)
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                dim_reduction_factor=model_args.dim_reduction_factor,
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                dim_reduction_factor=model_args.dim_reduction_factor,
            )
        return model

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        dim_reduction_factor: int,
        pooling: str = "cls",
        normalize: bool = False,
        lora_name_or_path: str = None,
        **hf_kwargs,
    ):

        base_model = cls.TRANSFORMER_CLS.from_pretrained(
            model_name_or_path,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            **hf_kwargs,
        )

        print(f"Model class: {base_model.__class__}")
        print(f"Dim reduction factor: {dim_reduction_factor}")

        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if lora_name_or_path:
            lora_config = LoraConfig.from_pretrained(lora_name_or_path, **hf_kwargs)
            lora_model = PeftModel.from_pretrained(
                base_model, lora_name_or_path, config=lora_config
            )
            lora_model = lora_model.merge_and_unload()
            model = cls(
                encoder=lora_model,
                dim_reduction_factor=dim_reduction_factor,
                pooling=pooling,
                normalize=normalize,
            )
        else:
            model = cls(
                encoder=base_model,
                dim_reduction_factor=dim_reduction_factor,
                pooling=pooling,
                normalize=normalize,
            )

        # Attempt to load the projection layer weights from a safetensors checkpoint
        checkpoint_path = os.path.join(model_name_or_path, "model.safetensors")
        if os.path.exists(checkpoint_path):
            checkpoint = load_safetensors(checkpoint_path, device="cpu")
            projection_keys = [
                "dim_reduct_linear_projection.weight",
                "dim_reduct_linear_projection.bias",
            ]
            if all(key in checkpoint for key in projection_keys):
                print(
                    "Loading dimensional reduction projection layer weights from safetensors..."
                )
                state_dict = {
                    key.replace("dim_reduct_linear_projection.", ""): checkpoint[key]
                    for key in projection_keys
                }
                model.encoder.dim_reduct_linear_projection.load_state_dict(state_dict)

            else:
                raise ValueError("No safetensors checkpoint found.")
        else:
            raise ValueError("No safetensors checkpoint found.")

        return model
