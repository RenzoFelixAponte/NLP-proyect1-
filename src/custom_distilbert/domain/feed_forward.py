"""MLP interno de cada bloque transformer (expande a 4x y vuelve a comprimir)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.custom_distilbert.domain.config import DistilBertConfig


class FeedForward(nn.Module):
    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return self.dropout(x)
