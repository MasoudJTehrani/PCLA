

from transformers import LlamaModel, LlamaConfig, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import GPTNeoXForCausalLM
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoModel, AutoTokenizer

from typing import Any, Dict, Optional, Tuple
from torch.nn import functional as F

import torch
from torch import Tensor, nn

CONFIGS: Dict[str, Dict[str, Any]] = {
    "debug": dict(num_hidden_layers=2, num_attention_heads=2, hidden_size=32, intermediate_size=64),
    "legacy-tiny": dict(num_hidden_layers=8, num_attention_heads=16, hidden_size=2048, intermediate_size=4096),
    # -- auto-regressive driving models --
    "tiny": dict(num_hidden_layers=12, num_attention_heads=8, hidden_size=512, intermediate_size=2048),  # 50M
    "x-small": dict(num_hidden_layers=14, num_attention_heads=8, hidden_size=1024, intermediate_size=4096),  # 235M
    "small": dict(num_hidden_layers=22, num_attention_heads=8, hidden_size=1024, intermediate_size=4096),  # 369M
    "medium": dict(num_hidden_layers=22, num_attention_heads=12, hidden_size=1536, intermediate_size=4096),  # 623M
    "large": dict(num_hidden_layers=22, num_attention_heads=16, hidden_size=2048, intermediate_size=5632),  # 1.1B
    # -- gaia world models --
    "gaia-large": dict(num_hidden_layers=22, num_attention_heads=16, hidden_size=1536, intermediate_size=4096),  # 623M
    # -- language models --
    "7B": dict(num_hidden_layers=32, num_attention_heads=32, hidden_size=4096, intermediate_size=11008),  # 6476M
    "13B": dict(num_hidden_layers=40, num_attention_heads=40, hidden_size=5120, intermediate_size=11008),
    "70B": dict(
        num_hidden_layers=80, num_attention_heads=64, hidden_size=8192, intermediate_size=28672, num_key_value_heads=8
    ),
    "tiny-llama-1.1b": dict(  # 969M
        num_hidden_layers=22, num_attention_heads=32, hidden_size=2048, intermediate_size=5632, num_key_value_heads=4
    ),
    "phi": dict(
        num_hidden_layers=24,
        num_attention_heads=32,
        hidden_size=2048,
        intermediate_size=8192,
        partial_rotary_factor=0.5,
        bias=True,
        vocab_size=51200,
        norm_type="layer_norm",
        mlp_type="gelu_new",
        parallel_attn_mlp=True,
    ),
}


class LLM(nn.Module):
    def __init__(self,
                **cfg,
            ):
        super().__init__()
        for key, value in cfg.items():
            setattr(self, key, value)

        if 'pythia' in self.variant:
            raise ValueError(f"Carefull: Variant {self.variant} not tested.")
            self.variant = f'EleutherAI/{self.variant}'
            self.model = GPTNeoXForCausalLM.from_pretrained(self.variant, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.variant, torch_dtype="auto",  trust_remote_code=True)
            self.model.embed_tokens = self.model.base_model.embed_in
        elif 'paligemma' in self.variant:
            raise ValueError(f"Carefull: Variant {self.variant} not tested.")
            self.variant = f'google/{self.variant}'
            from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                self.variant,
                torch_dtype="auto",
                revision="float16",
            ).language_model
            self.model.embed_tokens = self.model.base_model.embed_tokens
            self.tokenizer = AutoProcessor.from_pretrained(self.variant, torch_dtype="auto").tokenizer
        elif 'TinyLlama' in self.variant:
            raise ValueError(f"Carefull: Variant {self.variant} not tested.")
            print('Loading pretrained model')
            self.model = AutoModelForCausalLM.from_pretrained(self.variant, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.variant, torch_dtype="auto",  trust_remote_code=True)
            self.model.embed_tokens = self.model.base_model.embed_tokens
        elif 'llava-v1.6' in self.variant:
            raise ValueError(f"Carefull: Variant {self.variant} not tested.")
            print('Loading pretrained model')
            self.model = LlavaNextForConditionalGeneration.from_pretrained(self.variant, trust_remote_code=True)
            self.tokenizer = LlavaNextProcessor.from_pretrained(self.variant, torch_dtype="auto",  trust_remote_code=True).tokenizer
            self.model = self.model.language_model
            self.model.embed_tokens = self.model.base_model.embed_tokens
        elif 'internvl' in self.variant.lower():
            self.model = AutoModel.from_pretrained(self.variant, trust_remote_code=True)
            self.model = self.model.language_model
            try:
                self.model.embed_tokens = self.model.base_model.embed_tokens
            except:
                self.model.embed_tokens = self.model.model.tok_embeddings
        else:
            raise ValueError(f"Carefull: Variant {self.variant} not tested.")
            config_overrides = CONFIGS[self.variant].copy()
            configuration = LlamaConfig(**config_overrides)
            # Initializing a model from the llama-7b style configuration
            # self.model = LlamaForCausalLM(configuration)
            self.model = LlamaModel(configuration)
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
            self.model.embed_tokens = self.model.base_model.embed_tokens
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.lora:
            from peft import get_peft_model
            from peft import LoraConfig
            
            print('Using PEFT model')
            peft_config = LoraConfig(
                inference_mode=False, 
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules="all-linear",
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

        self.vocab_size = self.model.config.vocab_size
        self.hidden_size = self.model.config.hidden_size
        self.max_position_embeddings = self.model.config.max_position_embeddings


    def forward(self,
        embeddings: Tensor,
        attention_mask: Tensor = None,
        return_dict: bool = True,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:

        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True,
            position_ids=position_ids,
            return_dict=return_dict
        )#.last_hidden_state
        features = outputs.hidden_states[-1]
        logits = outputs[0]

        return features, logits

    def sample_categorical(
        self,
        logits: Tensor,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        restrict_tokens: Optional[Tuple[int, int]] = None,
    ):
        if restrict_tokens is not None:
            logits[..., : restrict_tokens[0]] = -float("inf")
            logits[..., restrict_tokens[0] + restrict_tokens[1] :] = -float("inf")

        if temperature <= 0.0:
            return logits.argmax(dim=-1, keepdim=False)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            pivot = v.select(-1, -1).unsqueeze(-1)
            logits = torch.where(logits < pivot, -float("inf"), logits)

        temperature = max(temperature, 1e-9)
        logits = logits / temperature

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            # Shift the indices to the right to keep also the first token above the threshold
            mask = (cumulative_probs > top_p).roll(shifts=1, dims=-1)
            mask[..., 0] = False
            logits[mask.gather(-1, sorted_indices.argsort(-1))] = -float("inf")

        return torch.multinomial(logits.softmax(dim=-1), 1).squeeze(-1)

    def greedy_sample(
        self,
        input_embeds: Tensor,
        inputs_mask: Optional[Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        cache_offset: int = 0,
        input_embed_matrix: Optional[Tensor] = None,
        logit_matrix: Optional[Tensor] = None,
        restrict_tokens: Optional[Tuple[int, int]] = None,
        attention_mask = None,
        position_ids = None,
    ) -> Tuple[Tensor, int]:
        
        if input_embed_matrix is None:
            if self.embed_tokens is None:
                raise ValueError(
                    "No input embeddings available because the model doesn't define a vocab. "
                    "Please provide input_embed_matrix. "
                )
            input_embed_matrix = self.embed_tokens.weight
        if logit_matrix is None:
            if self.lm_head is None:
                raise ValueError(
                    "No logit matrix available because the model doesn't define a vocab. "
                    "Please provide logit_matrix. "
                )
            logit_matrix = self.lm_head.weight
        # We generate tokens up to the eos token id if provided. If not provided, we generate until the end.
        # If no eos token id is provided, use -1 instead, so we will never stop generating until 'new_token'.
        sampled_tokens = torch.empty((input_embeds.size(0), max_new_tokens), device=input_embeds.device, dtype=torch.long)
        if eos_token_id is not None:
            sampled_tokens.fill_(eos_token_id)

        # we start with all sequences left to complete
        incomplete_seq_mask = torch.ones(input_embeds.size(0), dtype=torch.bool, device=input_embeds.device)
        for i in range(max_new_tokens):
            features, logits = self.forward(
                embeddings=input_embeds, 
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

            last_hidden_state = features[:, -1]

            # sample the next token
            logits = F.linear(last_hidden_state, logit_matrix)
            
            next_token = self.sample_categorical(
                logits, temperature=temperature, top_k=top_k, top_p=top_p, restrict_tokens=restrict_tokens
            )
            x = F.embedding(next_token.unsqueeze(1), input_embed_matrix)

            input_embeds = torch.cat([input_embeds, x], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((input_embeds.size(0), 1), device=input_embeds.device)], dim=1)

            # only update sequences where we haven't predicted the eos token before
            sampled_tokens[incomplete_seq_mask, i] = next_token[incomplete_seq_mask]
            # For completed sequences, we mask them out for future sampling
            # self.cache_mask[:, cache_offset - 1] &= incomplete_seq_mask

            if eos_token_id is not None:
                # only update the mask of incomplete sequences and stop early if an eos token id is provided.
                incomplete_seq_mask = sampled_tokens[:, i] != eos_token_id
                if not incomplete_seq_mask.any():
                    # finished all sequences, early exit
                    sampled_tokens = sampled_tokens[:, : i + 1]
                    break

        return sampled_tokens, input_embeds


if __name__ == "__main__":
    model = LLM("x-small", False)
    print(model)