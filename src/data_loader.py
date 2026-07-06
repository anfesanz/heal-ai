"""Data loading utilities for public NHANES files.

The preferred path for this portfolio project is to place a cleaned CSV in
`data/raw/nhanes_clean.csv`. The CDC XPT helpers below make the workflow
reproducible when internet access is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@dataclass(frozen=True)
class NHANESFile:
    name: str
    url: str
    columns: tuple[str, ...]


NHANES_2017_2020_FILES = (
    NHANESFile(
        "demographics",
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.xpt",
        ("SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2", "INDFMPIR"),
    ),
    NHANESFile(
        "body_measures",
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BMX_J.xpt",
        ("SEQN", "BMXBMI", "BMXWT", "BMXHT"),
    ),
    NHANESFile(
        "blood_pressure",
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BPX_J.xpt",
        ("SEQN", "BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4", "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"),
    ),
    NHANESFile(
        "diabetes",
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DIQ_J.xpt",
        ("SEQN", "DIQ010"),
    ),
    NHANESFile(
        "smoking",
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/SMQ_J.xpt",
        ("SEQN", "SMQ020"),
    ),
    NHANESFile(
        "cholesterol",
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/TCHOL_J.xpt",
        ("SEQN", "LBXTC"),
    ),
)


def read_clean_csv(path: str | Path = RAW_DIR / "nhanes_clean.csv") -> pd.DataFrame:
    """Read a user-provided cleaned NHANES CSV."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Add a cleaned NHANES CSV or call "
            "`load_or_build_nhanes_dataset(download=True)`."
        )
    return pd.read_csv(path)


def download_nhanes_xpt_files(
    files: Iterable[NHANESFile] = NHANES_2017_2020_FILES,
    raw_dir: str | Path = RAW_DIR,
) -> dict[str, Path]:
    """Download public CDC XPT files if they are not already present."""

    import requests

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for file in files:
        path = raw_dir / f"{file.name}.xpt"
        if not path.exists() or not _looks_like_xport(path):
            response = requests.get(file.url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
            if not _looks_like_xport(path):
                raise ValueError(f"Downloaded file from {file.url} is not a valid SAS XPORT file.")
        paths[file.name] = path
    return paths


def _looks_like_xport(path: Path) -> bool:
    """Return True when a cached file looks like a SAS XPORT file."""

    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as handle:
        header = handle.read(80)
    return b"HEADER RECORD*******LIBRARY HEADER RECORD" in header


def load_nhanes_xpt(paths: dict[str, str | Path]) -> pd.DataFrame:
    """Load and merge NHANES XPT files on participant id (`SEQN`)."""

    frames = []
    for name, path in paths.items():
        frame = pd.read_sas(path, format="xport")
        spec = next((item for item in NHANES_2017_2020_FILES if item.name == name), None)
        if spec is not None:
            columns = [col for col in spec.columns if col in frame.columns]
            frame = frame[columns]
        frames.append(frame)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="SEQN", how="left")
    return merged


def clean_nhanes(raw: pd.DataFrame) -> pd.DataFrame:
    """Create an analysis-ready dataset with a diabetes outcome."""

    df = raw.copy()
    df["age"] = df["RIDAGEYR"]
    df["sex"] = df["RIAGENDR"].map({1: "Male", 2: "Female"})
    df["race_ethnicity"] = df["RIDRETH3"].map(
        {
            1: "Mexican American",
            2: "Other Hispanic",
            3: "Non-Hispanic White",
            4: "Non-Hispanic Black",
            6: "Non-Hispanic Asian",
            7: "Other/Multi-racial",
        }
    )
    df["education"] = df["DMDEDUC2"].map(
        {
            1: "Less than 9th grade",
            2: "9th-11th grade",
            3: "High school/GED",
            4: "Some college/AA",
            5: "College graduate",
        }
    )
    df["poverty_group"] = pd.cut(
        df["INDFMPIR"],
        bins=[-np.inf, 1.0, 2.0, 4.0, np.inf],
        labels=["<=100% FPL", "101-200% FPL", "201-400% FPL", ">400% FPL"],
    ).astype("object")
    df["age_group"] = pd.cut(
        df["age"],
        bins=[17, 39, 59, np.inf],
        labels=["18-39", "40-59", "60+"],
    ).astype("object")
    df["bmi"] = df.get("BMXBMI")
    df["total_cholesterol"] = df.get("LBXTC")
    df["systolic_bp"] = df[[col for col in ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"] if col in df]].mean(axis=1)
    df["diastolic_bp"] = df[[col for col in ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"] if col in df]].mean(axis=1)
    df["ever_smoked_100_cigarettes"] = df["SMQ020"].map({1: "Yes", 2: "No"})
    # Borderline diabetes is treated as non-case for this demonstration outcome.
    df["diabetes"] = df["DIQ010"].map({1: 1, 2: 0, 3: 0})

    keep = [
        "SEQN",
        "diabetes",
        "age",
        "age_group",
        "sex",
        "race_ethnicity",
        "education",
        "poverty_group",
        "bmi",
        "systolic_bp",
        "diastolic_bp",
        "total_cholesterol",
        "ever_smoked_100_cigarettes",
    ]
    df = df[keep]
    df = df[df["age"].ge(18) & df["diabetes"].isin([0, 1])]
    return df.reset_index(drop=True)


def load_or_build_nhanes_dataset(
    clean_csv: str | Path = RAW_DIR / "nhanes_clean.csv",
    processed_csv: str | Path = PROCESSED_DIR / "nhanes_diabetes_analysis.csv",
    download: bool = False,
) -> pd.DataFrame:
    """Load a cleaned CSV, processed CSV, or build from public CDC files."""

    processed_csv = Path(processed_csv)
    if processed_csv.exists():
        return pd.read_csv(processed_csv)

    clean_csv = Path(clean_csv)
    if clean_csv.exists():
        df = read_clean_csv(clean_csv)
    elif download:
        paths = download_nhanes_xpt_files()
        df = clean_nhanes(load_nhanes_xpt(paths))
    else:
        raise FileNotFoundError(
            "No NHANES data found. Add `data/raw/nhanes_clean.csv` or run with "
            "`download=True` to retrieve public CDC files."
        )

    processed_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_csv, index=False)
    return df
