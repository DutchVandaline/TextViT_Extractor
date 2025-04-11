import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, ElectraModel

from SentenceDataset import SentenceDataset
from Train_Step import train_step
from Test_Step import test_step

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SEQ_LEN   = 128
BATCH_SIZE    = 256
NUM_EPOCHS    = 20
LEARNING_RATE = 1e-5

TRAIN_CSV = "/home/junha/TextViT_Extractor/Datasets/train_dataset_10percent.csv"
TEST_CSV  = "/home/junha/TextViT_Extractor/Datasets/test_dataset_10percent.csv"

tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-discriminator")

def collate_fn(batch):
    input_ids      = torch.stack([item[0] for item in batch], dim=0)  # (B,1,L)
    attention_mask = torch.stack([item[1] for item in batch], dim=0)  # (B,1,L)
    labels         = torch.stack([item[2] for item in batch], dim=0)  # (B,)
    return {
        'input_ids':      input_ids,
        'attention_mask': attention_mask,
        'label':          labels
    }

train_ds = SentenceDataset(TRAIN_CSV, tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN)
test_ds  = SentenceDataset(TEST_CSV,  tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

class KoElectraExtractor(nn.Module):
    def __init__(self, pretrained_model="monologg/koelectra-small-discriminator", dropout=0.1):
        super().__init__()
        self.electra = ElectraModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(dropout)
        hidden_size = self.electra.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x, attention_mask=None):
        B, S, L = x.size()
        input_ids = x.view(B * S, L)
        attn_mask = attention_mask.view(B * S, L) if attention_mask is not None else None

        outputs = self.electra(input_ids=input_ids, attention_mask=attn_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        cls_hidden = self.dropout(cls_hidden)
        logits = self.classifier(cls_hidden)  # (B*S, 1)
        return logits.view(B, S)              # (B, S)

model = KoElectraExtractor().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCEWithLogitsLoss()

best_f1 = 0.0
best_model_path = "best_kobert_f1.pt"

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_step(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc, f1 = test_step(model, test_loader, criterion, DEVICE)
    print(f"Epoch {epoch:02d} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f" Test Loss: {val_loss:.4f}, Test Acc: {val_acc:.4f}, F1: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), best_model_path)
        print(f"New best F1: {best_f1:.4f}. Model saved to {best_model_path}")
