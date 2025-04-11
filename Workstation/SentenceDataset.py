import pandas as pd
import torch
from torch.utils.data import Dataset

class SentenceDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_seq_len):
        self.df = pd.read_csv(csv_path, dtype={"doc_id": str}, low_memory=False)
        self.max_len = max_seq_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row["sentence"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        label = torch.tensor(row["label"], dtype=torch.float)

        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)

        return input_ids, attention_mask, label
