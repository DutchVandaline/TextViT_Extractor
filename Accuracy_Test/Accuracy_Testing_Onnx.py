import re
import numpy as np
import onnxruntime
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-discriminator")
onnx_model_path = "../Checkpoints/koelectra_extractor.onnx"
ort_session = onnxruntime.InferenceSession(onnx_model_path)

with open('Newspaper2.txt', 'r', encoding='utf-8') as f:
    news_article = f.read()

sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', news_article.strip()) if s.strip()]

threshold = 0.25
for sentence in sentences:
    encoded = tokenizer.encode_plus(sentence, max_length=128, truncation=True, padding='max_length', return_tensors="np")
    input_ids = encoded["input_ids"].reshape(1, 1, 128).astype(np.int64)
    attention_mask = encoded["attention_mask"].reshape(1, 1, 128).astype(np.int64)
    ort_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    ort_outs = ort_session.run(None, ort_inputs)
    logits = ort_outs[0]
    if logits.shape[-1] == 1:
        logits = np.squeeze(logits, axis=-1)
    probability = 1 / (1 + np.exp(-logits))
    prediction = (probability > threshold).astype(int)
    if prediction.item() == 1:
        print(sentence)
