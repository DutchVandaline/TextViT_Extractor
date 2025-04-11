import re
import sys
import torch
from transformers import AutoTokenizer
from Model.KoElectraExtractor import KoElectraExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KoElectraExtractor().to(device)
model_path = "../Checkpoints/best_kobert_f1.pt"
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()
tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-discriminator")

with open('Newspaper2.txt', 'r', encoding='utf-8') as f:
    news_article = f.read()

sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', news_article.strip()) if s.strip()]

for sentence in sentences:
    encoded = tokenizer.encode_plus(sentence, max_length=128, truncation=True, padding='max_length', return_tensors="pt")
    input_ids = encoded["input_ids"].unsqueeze(1).to(device)
    attention_mask = encoded["attention_mask"].unsqueeze(1).to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)
        probability = torch.sigmoid(logits)
        prediction = (probability > 0.25).int()
    if prediction.item() == 1:
        print(sentence)
