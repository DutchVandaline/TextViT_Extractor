import pandas as pd

#original_path =r"C:\junha\Datasets\Text_Extractor\Training\train_dataset.csv"
original_path =r"C:\junha\Datasets\Text_Extractor\Validation\test_dataset.csv"
output_path = r"C:\junha\Datasets\Text_Extractor\Validation\test_dataset_50percent.csv"

df = pd.read_csv(original_path, low_memory=False)
sample_df = df.sample(frac=0.5, random_state=42)
sample_df.to_csv(output_path, index=False)
