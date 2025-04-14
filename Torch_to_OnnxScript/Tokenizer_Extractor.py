from transformers import AutoTokenizer
import os

save_dir = "../Checkpoints/tokenizer"
os.makedirs(save_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(
    "monologg/koelectra-small-discriminator",
    use_fast=True
)

tokenizer.save_pretrained(save_dir)

print(f"Tokenizer saved to {save_dir}")
