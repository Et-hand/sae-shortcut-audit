"""Pythia-160m with a linear head on the last-token residual stream."""
import torch
from torch import nn
from transformers import AutoModel


class PythiaClassifier(nn.Module):
    def __init__(self, model_name: str = "EleutherAI/pythia-160m", num_labels: int = 2):
        super().__init__()
        self.body = AutoModel.from_pretrained(model_name)
        self.head = nn.Linear(self.body.config.hidden_size, num_labels)

    @staticmethod
    def last_token(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Right padding: last real token index = (number of real tokens) - 1.
        last = attention_mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.size(0), device=hidden.device), last]

    def forward(self, input_ids, attention_mask):
        hidden = self.body(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(self.last_token(hidden, attention_mask))
