"""The decision model: Ministral 3 text decoder, LoRA, delimiter embeddings and pointer head."""

import math

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from transformers import AutoTokenizer, DynamicCache, Mistral3ForConditionalGeneration

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


def sample_embeddings(weight: torch.Tensor, count: int) -> torch.Tensor:
    """Draw `count` vectors from a normal distribution with the mean and covariance of the trained embedding rows.

    A draw is mean + (X - mean)^T w / sqrt(n), with w ~ N(0, I_n). Its covariance is (X - mean)^T (X - mean) / n,
    so the hidden x hidden covariance matrix is not built. Rows with norm 0 were never trained and are not used.
    """
    rows = weight.detach().float()
    rows = rows[rows.norm(dim=-1) > 0]
    mean = rows.mean(dim=0)
    w = torch.randn(rows.shape[0], count, device=rows.device)
    return mean + ((rows - mean).T @ w).T / rows.shape[0] ** 0.5


def merged_decoder(decoder: nn.Module, adapter_dir, delimiter_ids: list[int]) -> nn.Module:
    """Merge the LoRA weights and the delimiter rows into the decoder: W' = W + (alpha / r) B A.

    Merge in the dtype of `decoder` (use fp32), then cast the result if necessary. One rounding step only.
    """
    peft = PeftModel.from_pretrained(decoder, str(adapter_dir))
    rows = next(p for name, p in peft.named_parameters() if "trainable_tokens_delta" in name).detach().clone()
    merged = peft.merge_and_unload()
    if not torch.equal(merged.get_input_embeddings().weight[delimiter_ids], rows):
        raise RuntimeError("the merge did not copy the trained delimiter rows into the embedding table")
    return merged


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
    def __init__(
        self, decoder: nn.Module, delimiter_ids: list[int], pad_id: int, adapter_dir=None, merge: bool = False
    ):
        """merge=True (inference only): load the adapter from adapter_dir and merge it into the decoder weights."""
        super().__init__()
        if merge:
            self.decoder = merged_decoder(decoder, adapter_dir, delimiter_ids)
        elif adapter_dir is None:
            decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=LORA_TARGETS,
                trainable_token_indices={"embed_tokens": delimiter_ids},
            )
            # The delimiter rows of Ministral 3 are all zeros. Start them as samples of the real embeddings instead.
            start = sample_embeddings(decoder.get_input_embeddings().weight, len(delimiter_ids))
            self.decoder = get_peft_model(decoder, config)
            with torch.no_grad():
                self.delimiter_rows().copy_(start)
        else:
            decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.decoder = PeftModel.from_pretrained(decoder, str(adapter_dir), is_trainable=True)
        self.head = PointerHead(decoder.config.hidden_size)
        self.pad_id = pad_id
        self.temperature = 1.0  # set by calibration; always 1.0 during training

    @property
    def device(self) -> torch.device:
        return self.head.q.weight.device

    def delimiter_rows(self) -> nn.Parameter:
        """The trainable embedding rows of the delimiters. PEFT uses them in place of the original rows."""
        return next(p for name, p in self.decoder.named_parameters() if "trainable_tokens_delta" in name)

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
        scores: list[torch.Tensor] = [torch.empty(0)] * len(rows)
        for batch in _passes(rows):
            for i, score in zip(batch, self.scores([rows[i] for i in batch]), strict=True):
                scores[i] = score.cpu()
        return scores

    @torch.no_grad()
    def predict_scores_cached(self, prefix: list[int], rows: list[Row]) -> list[torch.Tensor]:
        """predict_scores() for rows that all start with `prefix` (the state). The state is computed once; each row
        then runs only its question part on top of the cached keys and values of the state."""
        self.eval()
        size = len(prefix)
        if any(row.ids[:size] != prefix for row in rows):
            raise ValueError("every row must start with the prefix")
        autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda")
        with autocast:
            state = self.decoder(input_ids=torch.tensor([prefix], device=self.device), use_cache=True)
        cached = [(layer.keys, layer.values) for layer in state.past_key_values.layers]
        scores: list[torch.Tensor] = [torch.empty(0)] * len(rows)
        for batch in _passes([Row(r.ids[size:], 0, [], ()) for r in rows], prefix_len=size):
            n = len(batch)
            length = max(len(rows[i].ids) - size for i in batch)
            ids = torch.full((n, length), self.pad_id, dtype=torch.long)
            mask = torch.zeros((n, size + length), dtype=torch.long)
            mask[:, :size] = 1
            for j, i in enumerate(batch):
                question = rows[i].ids[size:]
                ids[j, : len(question)] = torch.tensor(question)
                mask[j, size : size + len(question)] = 1
            cache = DynamicCache(
                ddp_cache_data=[(k.expand(n, -1, -1, -1), v.expand(n, -1, -1, -1)) for k, v in cached],
                config=self.decoder.config,
            )
            positions = (size + torch.arange(length, device=self.device)).expand(n, length)
            with autocast:
                hidden = self.decoder(
                    input_ids=ids.to(self.device),
                    attention_mask=mask.to(self.device),
                    position_ids=positions,
                    past_key_values=cache,
                    use_cache=True,
                ).last_hidden_state.float()
            for j, i in enumerate(batch):
                row = rows[i]
                options = [index - size for index in row.option_indices]
                score = self.head(hidden[j, row.decide_index - size], hidden[j, options]) / self.temperature
                scores[i] = score.cpu()
        return scores


def _passes(rows: list[Row], prefix_len: int = 0) -> list[list[int]]:
    """Row indices in batches of similar length. A batch holds at most ROW_TOKENS_PER_PASS tokens, counting the
    state once for each row (a cached state is copied into each row of the batch)."""
    order = sorted(range(len(rows)), key=lambda i: len(rows[i].ids))
    batches, start = [], 0
    while start < len(order):
        end = start + 1
        while end < len(order) and (end - start + 1) * (prefix_len + len(rows[order[end]].ids)) <= ROW_TOKENS_PER_PASS:
            end += 1
        batches.append(order[start:end])
        start = end
    return batches


def delimiter_report(delimiter_ids: list[int]) -> str:
    """Compare the embedding norms of the delimiter tokens with the norms of ordinary tokens."""
    decoder = load_text_decoder(torch.bfloat16)
    norms = decoder.embed_tokens.weight.float().norm(dim=-1)
    ordinary = norms[1000:]
    lines = [f"ordinary tokens: mean norm {ordinary.mean():.4f}, std {ordinary.std():.4f}, min {ordinary.min():.4f}"]
    lines += [f"token {i}: norm {norms[i]:.4f}" for i in delimiter_ids]
    return "\n".join(lines)
