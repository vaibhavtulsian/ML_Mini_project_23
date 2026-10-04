import pandas as pd
from pathlib import Path
import subprocess


# ============================================================
# DOWNLOAD REQUIRED fMRI FILES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fmri_manifest.csv"
)

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ds007116"
)


print("=" * 70)
print("DOWNLOADING REQUIRED fMRI FILES")
print("=" * 70)


# ------------------------------------------------------------
# Load manifest
# ------------------------------------------------------------

manifest = pd.read_csv(MANIFEST_FILE)

# Only files that exist in the dataset structure
available = manifest[
    manifest["file_present"] == True
].copy()

print("\nTotal files to retrieve:", len(available))

print("\nTarget distribution:")
print(
    available["target"]
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# Confirm dataset directory
# ------------------------------------------------------------

print("\nDataLad dataset:")
print(DATASET_ROOT)


# ------------------------------------------------------------
# Download files
# ------------------------------------------------------------

success = 0
failed = []

for i, row in enumerate(
    available.itertuples(index=False),
    start=1
):

    relative_path = row.fmri_file

    print("\n" + "-" * 70)
    print(f"[{i}/{len(available)}]")
    print(relative_path)

    result = subprocess.run(
        [
            "datalad",
            "get",
            relative_path
        ],
        cwd=DATASET_ROOT
    )

    if result.returncode == 0:
        success += 1
        print("SUCCESS")
    else:
        failed.append(relative_path)
        print("FAILED")


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)

print("\nSuccessful:", success)
print("Failed:", len(failed))

if failed:
    print("\nFailed files:")

    for path in failed:
        print(path)