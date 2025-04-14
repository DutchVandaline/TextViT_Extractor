import torch

from Model.KoElectraExtractor import KoElectraExtractor

MAX_SEQ_LEN = 256
BATCH_SIZE = 1
S = 1

dummy_input_ids = torch.randint(0, 100, (BATCH_SIZE, S, MAX_SEQ_LEN))
dummy_attention_mask = torch.ones((BATCH_SIZE, S, MAX_SEQ_LEN), dtype=torch.long)

model = KoElectraExtractor()
model.load_state_dict(torch.load(r"../Checkpoints/50percent_best_kobert_f1.pt", map_location=torch.device("cpu")))
model.eval()

scripted_model = torch.jit.trace(model, (dummy_input_ids, dummy_attention_mask))
torch.jit.save(scripted_model, r"C:\junha\Git\TextViT_Extractor\Checkpoints\koelectra_extractor.pt")
print("모델이 성공적으로 TorchScript 형식으로 변환되어 저장되었습니다.")
