import torch
from Model.KoElectraExtractor import KoElectraExtractor
from transformers import AutoTokenizer

device = torch.device("cpu")
model = KoElectraExtractor().to(device)
model.load_state_dict(torch.load(r"C:\junha\Git\TextViT_Extractor\Checkpoints\50percent_best_kobert_f1.pt", map_location=device))
model.eval()

batch_size = 256
num_sentences = 1
seq_len = 256

tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-discriminator")

dummy_input = torch.randint(0, len(tokenizer), (batch_size, num_sentences, seq_len), dtype=torch.long)
dummy_attention = torch.ones((batch_size, num_sentences, seq_len), dtype=torch.long)

torch.onnx.export(
    model,
    (dummy_input, dummy_attention),
    "koelectra_extractor.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "num_sentences", 2: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "num_sentences", 2: "seq_len"},
        "logits": {0: "batch_size", 1: "num_sentences"}
    },
    opset_version=11,
)
print("ONNX 모델이 koelectra_extractor.onnx 로 저장되었습니다.")