import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from Model.TextViT_Extractor import SentenceExtractor
from Model.LSTM_Sentence import SentenceLSTMClassifier
from SentenceDataset import SentenceDataset
from Train_Step import train_step
from Test_Step import test_step

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SEQ_LEN = 128
BATCH_SIZE = 256
NUM_EPOCHS = 20
LEARNING_RATE = 5e-4

TRAIN_CSV = "/home/junha/TextViT_Extractor/Datasets/train_dataset_10percent.csv"
TEST_CSV  = "/home/junha/TextViT_Extractor/Datasets/test_dataset_10percent.csv"
tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-Medium", do_lower_case=False)

def collate_fn(batch):
    input_ids      = torch.stack([item[0] for item in batch], dim=0)  # (B,1,L)
    attention_mask = torch.stack([item[1] for item in batch], dim=0)  # (B,1,L)
    labels         = torch.stack([item[2] for item in batch], dim=0)  # (B,)
    return {
        'input_ids':      input_ids,
        'attention_mask': attention_mask,
        'label':          labels
    }


# Dataset & DataLoader
train_dataset = SentenceDataset(TRAIN_CSV, tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN)
test_dataset  = SentenceDataset(TEST_CSV,  tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

model = SentenceExtractor(
    vocab_size=tokenizer.vocab_size,
    max_seq_len=MAX_SEQ_LEN,
    num_transformer_layers=2
).to(DEVICE)

#model = SentenceLSTMClassifier(
#    vocab_size=tokenizer.vocab_size,
#    embedding_dim=300,
#    hidden_dim=256,
#    num_layers=2,
#    bidirectional=True,
#    dropout=0.3,
#    num_classes=1
#).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCEWithLogitsLoss()

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_step(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc, f1_score   = test_step(model, test_loader,  criterion, DEVICE)
    print(f"Epoch {epoch:02d} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f" Test Loss: {val_loss:.4f}, Test Acc: {val_acc:.4f}, F1 Score: {f1_score:.4f}")

