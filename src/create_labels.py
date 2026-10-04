import pandas as pd
from pathlib import Path


# ============================================================
# CREATE ALCOHOL-USE LABELS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

input_file = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ds007116"
    / "phenotype"
    / "substance.tsv"
)

output_file = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "alcohol_labels.csv"
)


print("=" * 70)
print("CREATING ALCOHOL-USE LABELS")
print("=" * 70)

print("\nLoading:")
print(input_file)


# ------------------------------------------------------------
# Load questionnaire
# ------------------------------------------------------------

df = pd.read_csv(input_file, sep="\t")

print("\nQuestionnaire loaded successfully.")
print("Shape:", df.shape)


# ------------------------------------------------------------
# Keep the variables we need
# ------------------------------------------------------------

label_df = df[
    [
        "participant_id",
        "session_id",
        "substance_alc_010"
    ]
].copy()


# ------------------------------------------------------------
# Remove missing alcohol responses
# ------------------------------------------------------------

label_df = label_df.dropna(
    subset=["substance_alc_010"]
).copy()


# ------------------------------------------------------------
# Create binary target
#
# 0 = No alcohol use in past 12 months
#     questionnaire values 1, 2, 3
#
# 1 = Alcohol use in past 12 months
#     questionnaire values 4, 5
# ------------------------------------------------------------

label_df["target"] = label_df["substance_alc_010"].apply(
    lambda x: 1 if x in [4, 5] else 0
)


# ------------------------------------------------------------
# Rename columns for clarity
# ------------------------------------------------------------

label_df = label_df.rename(
    columns={
        "participant_id": "subject",
        "session_id": "session"
    }
)


# ------------------------------------------------------------
# Keep only the columns needed for ML
# ------------------------------------------------------------

label_df = label_df[
    [
        "subject",
        "session",
        "target"
    ]
]


# ------------------------------------------------------------
# Create output directory
# ------------------------------------------------------------

output_file.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Save labels
# ------------------------------------------------------------

label_df.to_csv(
    output_file,
    index=False
)


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LABEL CREATION COMPLETE")
print("=" * 70)

print("\nSaved to:")
print(output_file)

print("\nNumber of labeled sessions:")
print(len(label_df))

print("\nTarget distribution:")
print(label_df["target"].value_counts().sort_index())

print("\nTarget percentages:")
print(
    label_df["target"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)

print("\nFirst 20 labels:")
print(label_df.head(20).to_string(index=False))

print("\nUnique subjects:")
print(label_df["subject"].nunique())

print("\nUnique sessions:")
print(label_df["session"].nunique())