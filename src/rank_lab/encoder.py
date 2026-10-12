"""MiniLM mean pooling with explicit local-only loading and sequence limits."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional
from transformers import AutoModel, AutoTokenizer

from rank_lab.artifacts import MODEL_ID


class Encoder(nn.Module):
    def __init__(self, directory: Path, *, device: str, max_length: int = 256) -> None:
        super().__init__()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModel.from_pretrained(
            directory,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            attn_implementation="eager",
        )
        self.to(device)

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(device)
        hidden = self.model(**tokens).last_hidden_state.float()
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return functional.normalize(pooled, p=2, dim=1)

    def encode(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        self.eval()
        rows = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                rows.append(self(texts[start : start + batch_size]).cpu().numpy())
        return np.concatenate(rows) if rows else np.empty((0, self.model.config.hidden_size))

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.model.config._name_or_path = MODEL_ID
        self.model.save_pretrained(directory, safe_serialization=True)
        self.tokenizer.name_or_path = MODEL_ID
        self.tokenizer.init_kwargs["name_or_path"] = MODEL_ID
        self.tokenizer.save_pretrained(directory)
