import torch.nn as nn
import torch
from transformers import ElectraModel


class KoElectraExtractor(nn.Module):
    def __init__(self, pretrained_model="monologg/koelectra-small-discriminator", dropout=0.1):
        super().__init__()
        self.electra = ElectraModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(dropout)
        hidden_size = self.electra.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x, attention_mask=None):
        # x: (B, num_sentences, seq_len)
        B, S, L = x.size()
        input_ids = x.view(B * S, L)
        attn_mask = attention_mask.view(B * S, L) if attention_mask is not None else None

        outputs = self.electra(input_ids=input_ids, attention_mask=attn_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        cls_hidden = self.dropout(cls_hidden)
        logits = self.classifier(cls_hidden)             # (B*S, 1)
        return logits.view(B, S)                         # (B, S)
