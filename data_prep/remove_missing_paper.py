"""
Keeps only the rows whose image file exists on disk.

Adapted from the original script `remove_missing_files.py` of the paper repository (cfredinh/bioactive, Apache-2.0).
Differences from the original (details in docs/UPSTREAM_COMPARISON.md):
  - drops rows without an image path first, then keeps rows whose file exists (same result)
  - prints per-fold counts; paths come from BIOACTIVE_DATA_DIR
Paths: set BIOACTIVE_DATA_DIR (default ./data) to the folder that holds the inputs and receives the outputs.
"""
import os
import pandas as pd
import os
_D = os.environ.get("BIOACTIVE_DATA_DIR", "data").rstrip("/") + "/"   # folder with metadata/, images/ and the CSVs

DF_PATH   = _D + "split_data_paper.csv"
BASE_PATH = _D + "images/"
OUT_CSV   = _D + "training_paper.csv"

df = pd.read_csv(DF_PATH, index_col=0)
print("Total rows:", len(df))
print("Rows with NaN Metadata_Path:", df.Metadata_Path.isna().sum())
df = df.dropna(subset=["Metadata_Path"])

present, missing = [], 0
for i, row in df.iterrows():
    if os.path.isfile(BASE_PATH + str(row.Metadata_Path)):
        present.append(i)
    else:
        missing += 1

print("Present:", len(present), " Missing:", missing)
out = df.loc[present]
out.to_csv(OUT_CSV)
print("Wrote", OUT_CSV, "shape", out.shape)

# per-fold sanity
print("\nPer-fold rows kept:")
print(out.split_number.value_counts().sort_index())
