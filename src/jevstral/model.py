"""The decision model: Ministral 3 text decoder, LoRA, delimiter embeddings and pointer head."""

import math

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration

from .encode import Row

BASE_MODEL = "mistralai/Ministral-3-8B-Base-2512"
BASE_REVISION = "d4883f9b36aa2e5d775730d3fdba3d30de51a8ef"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
POINTER_DIM = 256
ROW_TOKENS_PER_PASS = 32768  # inference: the token budget of one batch of rows


def load_tokenizer():
    return AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)


def load_text_decoder(dtype: torch.dtype = torch.float32) -> nn.Module:
    """Load the full image-text checkpoint and keep only its text decoder.

    Do not use Ministral3Model.from_pretrained on this repository: it does not map the
    checkpoint names, and it gives random weights without an error.
    """
    full = Mistral3ForConditionalGeneration.from_pretrained(BASE_MODEL, revision=BASE_REVISION, dtype=torch.bfloat16)
    decoder = full.model.language_model
    del full
    decoder.config.use_cache = False
    return decoder.to(dtype)


class PointerHead(nn.Module):
    """Score each option: the dot product of q(<decide>) and k(</opt>)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.q = nn.Linear(hidden_size, POINTER_DIM)
        self.k = nn.Linear(hidden_size, POINTER_DIM)

    def forward(self, decide: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        # decide: [hidden], options: [K, hidden] -> scores: [K]
        return self.k(options) @ self.q(decide) / math.sqrt(POINTER_DIM)


class DecisionModel(nn.Module):
    def __init__(self, decoder: nn.Module, delimiter_ids: list[int], pad_id: int, adapter_dir=None):
        super().__init__()
        decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if adapter_dir is None:
            config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=LORA_TARGETS,
                trainable_token_indices={"embed_tokens": delimiter_ids},
            )
            self.decoder = get_peft_model(decoder, config)
        else:
            self.decoder = PeftModel.from_pretrained(decoder, str(adapter_dir), is_trainable=True)
        self.head = PointerHead(decoder.config.hidden_size)
        self.pad_id = pad_id
        self.temperature = 1.0  # set by calibration; always 1.0 during training

    @property
    def device(self) -> torch.device:
        return self.head.q.weight.device

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def scores(self, rows: list[Row]) -> list[torch.Tensor]:
        """Run the rows as one right-padded batch. Return the option scores of each row, divided by T."""
        length = max(len(row.ids) for row in rows)
        ids = torch.full((len(rows), length), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), length), dtype=torch.long)
        for i, row in enumerate(rows):
            ids[i, : len(row.ids)] = torch.tensor(row.ids)
            mask[i, : len(row.ids)] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            hidden = self.decoder(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).last_hidden_state
        hidden = hidden.float()
        return [
            self.head(hidden[i, row.decide_index], hidden[i, row.option_indices]) / self.temperature
            for i, row in enumerate(rows)
        ]

    @torch.no_grad()
    def predict_scores(self, rows: list[Row]) -> list[torch.Tensor]:
        """Option scores (divided by T) for each row, in the input order. Rows go in batches of similar length."""
        self.eval()
        order = sorted(range(len(rows)), key=lambda i: len(rows[i].ids))
        scores: list[torch.Tensor] = [torch.empty(0)] * len(rows)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and (end - start + 1) * len(rows[order[end]].ids) <= ROW_TOKENS_PER_PASS:
                end += 1
            batch = order[start:end]
            for i, score in zip(batch, self.scores([rows[i] for i in batch]), strict=True):
                scores[i] = score.cpu()
            start = end
        return scores


def delimiter_report(delimiter_ids: list[int]) -> str:
    """Compare the embedding norms of the delimiter tokens with the norms of ordinary tokens."""
    decoder = load_text_decoder(torch.bfloat16)
    norms = decoder.embed_tokens.weight.float().norm(dim=-1)
    ordinary = norms[1000:]
    lines = [f"ordinary tokens: mean norm {ordinary.mean():.4f}, std {ordinary.std():.4f}, min {ordinary.min():.4f}"]
    lines += [f"token {i}: norm {norms[i]:.4f}" for i in delimiter_ids]
    return "\n".join(lines)
