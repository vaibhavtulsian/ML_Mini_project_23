from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATHS
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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_dataset.csv"
)


# ============================================================
# LOAD AAL ATLAS
# ============================================================

print("=" * 60)
print("LOADING AAL ATLAS")
print("=" * 60)

atlas = datasets.fetch_atlas_aal()
atlas_file = atlas.maps

print("Atlas:", atlas_file)

masker = NiftiLabelsMasker(
    labels_img=atlas_file,
    standardize=True
)


# ============================================================
# LOAD MANIFEST
# ============================================================

print("\nLoading manifest...")

manifest = pd.read_csv(MANIFEST_FILE)

print("Total manifest entries:", len(manifest))


# ============================================================
# SELECT DOWNLOADED fMRI FILES
# ============================================================

downloaded = []

for _, row in manifest.iterrows():

    fmri_path = DATASET_ROOT / row["fmri_file"]

    if fmri_path.is_file() and fmri_path.stat().st_size > 1_000_000:
        downloaded.append(row)


downloaded_df = pd.DataFrame(downloaded)

print("Downloaded fMRI sessions:", len(downloaded_df))

if len(downloaded_df) == 0:
    raise RuntimeError("No downloaded fMRI files found.")


# ============================================================
# EXTRACT FEATURES
# ============================================================

all_features = []

total = len(downloaded_df)

for index, row in downloaded_df.iterrows():

    subject = row["subject"]
    session = row["session"]
    target = int(row["target"])

    fmri_path = DATASET_ROOT / row["fmri_file"]

    print("\n" + "=" * 60)
    print(f"Processing {index + 1}/{total}")
    print("Subject:", subject)
    print("Session:", session)
    print("Target:", target)
    print("File:", fmri_path)

    try:

        # Extract ROI time series
        time_series = masker.fit_transform(str(fmri_path))

        print("Time series shape:", time_series.shape)

        # Dynamic range for every ROI
        dynamic_range = (
            time_series.max(axis=0)
            - time_series.min(axis=0)
        )

        print("Number of ROI features:", len(dynamic_range))

        # Create row
        data = {
            "subject": subject,
            "session": session,
            "target": target
        }

        for i, value in enumerate(dynamic_range):
            data[f"ROI_{i + 1}"] = value

        all_features.append(data)

    except Exception as e:

        print("ERROR processing file:")
        print(e)

        continue


# ============================================================
# CREATE DATAFRAME
# ============================================================

if len(all_features) == 0:
    raise RuntimeError("No fMRI files were successfully processed.")


df = pd.DataFrame(all_features)


# ============================================================
# SAVE DATASET
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
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 60)

print("Successful sessions:", len(df))
print("Unique subjects:", df["subject"].nunique())
print("Number of columns:", len(df.columns))
print("Number of ROI features:", len(df.columns) - 3)

print("\nTarget distribution:")
print(df["target"].value_counts().sort_index())

print("\nSaved to:")
print(OUTPUT_FILE)