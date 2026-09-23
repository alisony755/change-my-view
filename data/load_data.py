from datasets import load_dataset

ds = load_dataset("stanfordnlp/SHP")

df = ds["train"].to_pandas()
print(df.head())