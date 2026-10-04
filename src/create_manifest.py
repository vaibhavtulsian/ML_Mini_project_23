import pandas as pd
from pathlib import Path
import os


# ============================================================
# CREATE fMRI MANIFEST
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LABEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "alcohol_labels.csv"
)

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ds007116"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fmri_manifest.csv"
)


print("=" * 70)
print("CREATING fMRI MANIFEST")
print("=" * 70)


# ------------------------------------------------------------
# Load alcohol labels
# ------------------------------------------------------------

print("\nLoading labels:")
print(LABEL_FILE)

labels = pd.read_csv(LABEL_FILE)

print("Labels loaded.")
print("Number of labeled sessions:", len(labels))


# ------------------------------------------------------------
# Find resting-state fMRI
# ------------------------------------------------------------

manifest = []

for _, row in labels.iterrows():

    subject = row["subject"]
    session = row["session"]
    target = row["target"]

    # We use resting-state run-01
    fmri_relative = (
        Path(subject)
        / session
        / "func"
        / f"{subject}_{session}_task-rest_run-01_bold.nii.gz"
    )

    fmri_absolute = DATASET_ROOT / fmri_relative

    # lexists() also detects DataLad/Git-annex symlinks
    file_exists = os.path.lexists(fmri_absolute)

    manifest.append({
        "subject": subject,
        "session": session,
        "target": target,
        "fmri_file": str(fmri_relative),
        "file_present": file_exists
    })


# ------------------------------------------------------------
# Create DataFrame
# ------------------------------------------------------------

manifest_df = pd.DataFrame(manifest)


# ------------------------------------------------------------
# Save manifest
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

manifest_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# Print summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MANIFEST CREATED")
print("=" * 70)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nTotal labeled sessions:")
print(len(manifest_df))

print("\nSessions with a corresponding fMRI file entry:")
print(manifest_df["file_present"].sum())

print("\nSessions without an fMRI file entry:")
print((~manifest_df["file_present"]).sum())

print("\nTarget distribution:")
print(
    manifest_df["target"]
    .value_counts()
    .sort_index()
)

print("\nFirst 20 rows:")
print(
    manifest_df
    .head(20)
    .to_string(index=False)
)