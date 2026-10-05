# Predicting Alcohol Use from Resting-State fMRI Using Machine Learning

**CS229 — Machine Learning Mini Project (Team 23)**

## Overview

This project investigates whether resting-state functional MRI (fMRI) brain activity patterns can predict alcohol use behaviour. We extract region-of-interest (ROI) features from the AAL (Automated Anatomical Labeling) brain atlas, train several classical ML models, and evaluate them with subject-aware cross-validation to prevent data leakage.

---

## Dataset

| Item | Detail |
|---|---|
| **Source** | [OpenNeuro ds007116](https://openneuro.org/datasets/ds007116) |
| **Modality** | Resting-state fMRI (`task-rest`, `run-01`) |
| **Labelling variable** | `substance_alc_010` from `phenotype/substance.tsv` |
| **Binary target** | **0** — No alcohol use in past 12 months (values 1–3) · **1** — Alcohol use in past 12 months (values 4–5) |
| **Labelled sessions** | 133 |
| **Successfully processed sessions** | 75 (after download + feature extraction) |
| **Class distribution** | 63 negative (0) · 12 positive (1) — **imbalanced** |
| **Unique subjects** | Multiple (some subjects have >1 session) |

---

## Project Structure

```
cs229_alcohol_ml/
├── data/
│   ├── raw/
│   │   └── ds007116/            # OpenNeuro dataset (DataLad clone)
│   └── processed/
│       ├── alcohol_labels.csv   # Binary alcohol-use labels per session
│       ├── fmri_manifest.csv    # Mapping of sessions → fMRI file paths
│       ├── ml_dataset.csv       # Final ML-ready dataset (ROI features)
│       └── ml_dataset_demographics.csv  # ML dataset + age & sex
│
├── src/
│   ├── create_labels.py         # Step 1: Create binary labels from questionnaire
│   ├── create_manifest.py       # Step 2: Map sessions to fMRI file paths
│   ├── download_fmri.py         # Step 3: Download fMRI files via DataLad
│   ├── extract_features.py      # Step 4: Extract ROI features using AAL atlas
│   ├── add_demographics.py      # Step 5: Merge age & sex from participants.tsv
│   ├── train.py                 # Step 6: Baseline model comparison (fMRI only)
│   ├── train_improved.py        # Step 7: Compare fMRI-only vs fMRI+demographics
│   ├── pca.py                   # Step 8: PCA dimensionality reduction experiment
│   ├── tune_model.py            # Step 9: GridSearchCV for RBF SVM hyperparameters
│   ├── final_evaluation.py      # Step 10: Final tuned pipeline evaluation + model saving
│   └── plot_results.py          # Step 11: Confusion matrix & ROC curve plots
│
├── models/
│   ├── final_svm_pipeline.joblib   # Saved final pipeline (Imputer→Scaler→PCA→SVM)
│   └── feature_columns.json        # Ordered list of feature column names
│
├── results/
│   ├── model_comparison.csv              # Baseline model results
│   ├── improved_model_comparison.csv     # fMRI vs fMRI+demographics results
│   ├── pca_model_comparison.csv          # PCA experiment results
│   ├── svm_tuning_results.csv            # Full GridSearchCV output
│   ├── final_cv_results.csv              # Per-fold results of the tuned model
│   ├── cv_predictions.csv                # All CV predictions and probabilities
│   ├── confusion_matrix.png              # Confusion matrix plot
│   └── roc_curve.png                     # ROC curve plot
│
├── venv/                        # Python virtual environment
├── .gitignore
└── README.md
```

---

## Pipeline — Step by Step

### Step 1 · Create Labels (`create_labels.py`)

- Reads the substance-use questionnaire from `phenotype/substance.tsv`.
- Extracts the `substance_alc_010` field (alcohol use frequency in the past 12 months).
- Creates a binary target:
  - **0** → questionnaire values 1, 2, 3 (no or minimal alcohol use)
  - **1** → questionnaire values 4, 5 (regular alcohol use)
- Saves `alcohol_labels.csv` with columns: `subject`, `session`, `target`.

### Step 2 · Create Manifest (`create_manifest.py`)

- Reads `alcohol_labels.csv`.
- For each labelled session, constructs the expected resting-state fMRI file path (`task-rest_run-01_bold.nii.gz`).
- Checks whether the file exists in the DataLad dataset (via `os.path.lexists`).
- Saves `fmri_manifest.csv` with columns: `subject`, `session`, `target`, `fmri_file`, `file_present`.

### Step 3 · Download fMRI Files (`download_fmri.py`)

- Reads the manifest and filters to sessions where a file entry exists.
- Downloads each NIfTI file using `datalad get`.
- Tracks success/failure counts.

### Step 4 · Extract Features (`extract_features.py`)

- Loads the **AAL atlas** (116 regions) via `nilearn.datasets.fetch_atlas_aal`.
- Creates a `NiftiLabelsMasker` to extract standardised ROI time series from each fMRI scan.
- For each downloaded session:
  - Extracts the full ROI time series (time points × 116 regions).
  - Computes the **dynamic range** (max − min) per ROI as the feature vector.
- Produces `ml_dataset.csv` with 166 ROI feature columns + `subject`, `session`, `target`.

### Step 5 · Add Demographics (`add_demographics.py`)

- Reads `participants.tsv` from the raw dataset.
- Extracts `age` and `sex` per subject.
- Merges with the ML dataset on `subject`.
- Encodes sex as numeric (M=0, F=1).
- Saves `ml_dataset_demographics.csv`.

### Step 6 · Baseline Training (`train.py`)

Trains 5 models using **5-fold StratifiedGroupKFold** (subjects never split across folds):

| Model | Description |
|---|---|
| Logistic Regression | `class_weight="balanced"`, `max_iter=5000` |
| Linear SVM | `kernel="linear"`, `class_weight="balanced"` |
| RBF SVM | `kernel="rbf"`, `class_weight="balanced"` |
| Polynomial SVM | `kernel="poly"`, `degree=3`, `class_weight="balanced"` |
| MLP | `hidden_layer_sizes=(32,16)`, `early_stopping=True` |

Each pipeline: `SimpleImputer(median)` → `StandardScaler` → Classifier

**Metrics**: Balanced Accuracy, Precision, Recall, F1, ROC-AUC

### Step 7 · fMRI vs fMRI + Demographics (`train_improved.py`)

- Runs the same 5 models on two feature sets:
  1. **fMRI Only** — 166 ROI features
  2. **fMRI + Demographics** — 166 ROI features + age + sex
- Compares whether demographics improve classification.

### Step 8 · PCA Experiment (`pca.py`)

- Tests PCA with **n_components ∈ {5, 10, 15, 20, 30}** on both feature sets.
- Evaluates 3 models (Logistic Regression, Linear SVM, RBF SVM) across all PCA settings.
- Pipeline: `SimpleImputer` → `StandardScaler` → `PCA` → Classifier

### Step 9 · Hyperparameter Tuning (`tune_model.py`)

- Performs **GridSearchCV** on the RBF SVM pipeline.
- Search space:

| Parameter | Values |
|---|---|
| `pca__n_components` | 5, 10, 15, 20, 30 |
| `model__C` | 0.1, 0.5, 1, 2, 5, 10 |
| `model__gamma` | `"scale"`, 0.001, 0.01, 0.1 |
| `model__class_weight` | `"balanced"`, `None` |

- Scoring metric: **ROC-AUC**
- CV: 5-fold `StratifiedGroupKFold`
- Saves full results to `svm_tuning_results.csv`.

### Step 10 · Final Evaluation (`final_evaluation.py`)

- Builds the tuned pipeline with best parameters:
  - **PCA**: 5 components
  - **SVM**: `C=5`, `gamma=0.001`, `class_weight="balanced"`
- Runs 5-fold subject-aware CV to report unbiased metrics.
- Trains the final model on **all data** and saves:
  - `models/final_svm_pipeline.joblib`
  - `models/feature_columns.json`
  - `results/final_cv_results.csv`

### Step 11 · Plotting (`plot_results.py`)

- Generates a **confusion matrix** from aggregated CV predictions.
- Generates a **ROC curve** with AUC annotation.
- Saves plots to `results/confusion_matrix.png` and `results/roc_curve.png`.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **StratifiedGroupKFold** | Prevents data leakage — all scans from the same subject stay in one fold |
| **`class_weight="balanced"`** | Compensates for the 63:12 class imbalance |
| **Dynamic range as feature** | Simple, interpretable summary of ROI activity over the resting-state scan |
| **PCA** | Reduces 166 ROI features to a lower-dimensional space; mitigates overfitting on small data |
| **Pipeline architecture** | Imputation, scaling, PCA, and classification are all inside the pipeline to prevent train–test leakage |

---

## Results Summary

### Baseline Model Comparison (fMRI Only)

| Model | Balanced Acc | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| MLP | 0.471 | 0.105 | 0.467 | 0.154 | 0.212 |
| Polynomial SVM | 0.466 | 0.085 | 0.133 | 0.095 | 0.569 |
| RBF SVM | 0.478 | 0.067 | 0.067 | 0.067 | 0.480 |
| Logistic Regression | 0.447 | 0.040 | 0.067 | 0.050 | 0.533 |
| Linear SVM | 0.431 | 0.033 | 0.067 | 0.044 | 0.488 |

### Tuned RBF SVM (Final Model — PCA 5, C=5, γ=0.001)

| Fold | Balanced Acc | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| 1 | 0.603 | 0.250 | 0.667 | 0.364 | 0.538 |
| 2 | 0.577 | 0.214 | 1.000 | 0.353 | 0.256 |
| 3 | 0.292 | 0.000 | 0.000 | 0.000 | 0.625 |
| 4 | 0.750 | 0.250 | 1.000 | 0.400 | 0.292 |
| 5 | 0.519 | 0.143 | 0.500 | 0.222 | 0.500 |
| **Mean ± Std** | **0.548 ± 0.153** | **0.171 ± 0.094** | **0.633 ± 0.378** | **0.268 ± 0.145** | **0.442 ± 0.139** |

---

## How to Run

### Prerequisites

- Python 3.8+
- [DataLad](https://www.datalad.org/) (for downloading the raw fMRI data)

### Setup

```bash
# Clone the repo
git clone git@github.com:vaibhavtulsian/ML_MINI_PROJECT_TEAM_23.git
cd ML_MINI_PROJECT_TEAM_23

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install pandas numpy scikit-learn nilearn nibabel matplotlib joblib
```

### Run the Full Pipeline

```bash
# 1. Create binary labels from questionnaire
python src/create_labels.py

# 2. Build fMRI file manifest
python src/create_manifest.py

# 3. Download fMRI files via DataLad
python src/download_fmri.py

# 4. Extract ROI features using AAL atlas
python src/extract_features.py

# 5. Add demographic features (age, sex)
python src/add_demographics.py

# 6. Baseline model comparison
python src/train.py

# 7. Compare fMRI-only vs fMRI+demographics
python src/train_improved.py

# 8. PCA dimensionality reduction experiment
python src/pca.py

# 9. Hyperparameter tuning (GridSearchCV)
python src/tune_model.py

# 10. Final evaluation and model saving
python src/final_evaluation.py

# 11. Generate confusion matrix and ROC curve
python src/plot_results.py
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `nilearn` | fMRI data loading, atlas, ROI extraction |
| `nibabel` | NIfTI file handling (nilearn dependency) |
| `matplotlib` | Plotting (confusion matrix, ROC curve) |
| `joblib` | Model serialisation |
| `datalad` | fMRI data download from OpenNeuro |

---

## Authors

**Team 23** — CS229 Machine Learning Mini Project
