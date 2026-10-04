import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ML_DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_dataset.csv"
)

PARTICIPANTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ds007116"
    / "participants.tsv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_dataset_demographics.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("ADDING DEMOGRAPHIC FEATURES")
print("=" * 60)

ml_df = pd.read_csv(ML_DATASET)

participants = pd.read_csv(
    PARTICIPANTS_FILE,
    sep="\t"
)

print("ML dataset:", ml_df.shape)
print("Participants:", participants.shape)


# ============================================================
# SELECT DEMOGRAPHICS
# ============================================================

demo = participants[
    [
        "participant_id",
        "age",
        "sex"
    ]
].copy()

demo = demo.rename(
    columns={
        "participant_id": "subject"
    }
)


# ============================================================
# MERGE
# ============================================================

df = ml_df.merge(
    demo,
    on="subject",
    how="left"
)


# ============================================================
# ENCODE SEX
# ============================================================

df["sex"] = df["sex"].map(
    {
        "M": 0,
        "F": 1
    }
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DEMOGRAPHIC DATASET CREATED")
print("=" * 60)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nAge missing:", df["age"].isna().sum())
print("Sex missing:", df["sex"].isna().sum())

print("\nTarget distribution:")
print(df["target"].value_counts().sort_index())

print("\nSubjects:", df["subject"].nunique())

print("\nSaved to:")
print(OUTPUT_FILE)