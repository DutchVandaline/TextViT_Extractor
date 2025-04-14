import os
import numpy as np
import onnxruntime
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from SentenceDataset import SentenceDataset
from sklearn.metrics import accuracy_score, f1_score

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
TEST_CSV = "/home/junha/TextViT_Extractor/Datasets/test_dataset_10percent.csv"
MAX_SEQ_LEN = 256
BATCH_SIZE = 256

tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-discriminator")


def collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch], dim=0)
    attention_mask = torch.stack([item[1] for item in batch], dim=0)
    labels = torch.stack([item[2] for item in batch], dim=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "label": labels}


test_ds = SentenceDataset(TEST_CSV, tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

onnx_model_path = "koelectra_extractor.onnx"
ort_session = onnxruntime.InferenceSession(onnx_model_path)

all_preds = []
all_labels = []

for batch in test_loader:
    input_ids_np = batch["input_ids"].numpy()
    attention_mask_np = batch["attention_mask"].numpy()
    labels_np = batch["label"].numpy()

    ort_inputs = {"input_ids": input_ids_np, "attention_mask": attention_mask_np}
    ort_outs = ort_session.run(None, ort_inputs)
    logits = ort_outs[0]

    if logits.shape[-1] == 1:
        logits = np.squeeze(logits, axis=-1)

    probabilities = 1 / (1 + np.exp(-logits))
    preds_binary = (probabilities > 0.5).astype(int)

    all_preds.extend(preds_binary.tolist())
    all_labels.extend(labels_np.tolist())

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print("Test Accuracy: {:.4f}".format(acc))
print("Test F1 Score: {:.4f}".format(f1))
