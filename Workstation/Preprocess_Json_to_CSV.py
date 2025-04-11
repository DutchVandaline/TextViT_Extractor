import os
import json
import pandas as pd

JSON_DIR = r"C:\junha\Datasets\Text_Extractor\Validation"
OUTPUT_CSV = r"C:\junha\Datasets\Text_Extractor\Validation\test_dataset.csv"

def json_folder_to_csv(json_dir: str, output_csv: str):
    rows = []
    for fname in os.listdir(json_dir):
        if not fname.lower().endswith(".json"):
            continue
        path = os.path.join(json_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for doc in data.get("documents", []):
            extractive_set = set(doc.get("extractive", []))
            for paragraph in doc.get("text", []):
                for sen in paragraph:
                    rows.append({
                        "doc_id": doc.get("id"),
                        "sentence_index": sen.get("index"),
                        "sentence": sen.get("sentence"),
                        "label": 1 if sen.get("index") in extractive_set else 0
                    })
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"완료: {output_csv} ({len(df)} rows)")

if __name__ == "__main__":
    json_folder_to_csv(JSON_DIR, OUTPUT_CSV)
