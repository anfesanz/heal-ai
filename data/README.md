# Data

This project uses public NHANES data from the CDC.

Recommended workflow:

1. Put raw or cleaned CSV files in `data/raw/`.
2. Use `src.data_loader.load_or_build_nhanes_dataset()` to load a cleaned file or build one from CDC XPT files.
3. Save analysis-ready files to `data/processed/`.

Raw data files are intentionally ignored by git because NHANES files can be downloaded from public CDC URLs and may be large.
