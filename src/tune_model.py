import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import (
    StratifiedGroupKFold,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_dataset_demographics.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

roi_columns = [
    c for c in df.columns
    if c.startswith("ROI_")
]

X = df[roi_columns].copy()
y = df["target"].astype(int)
groups = df["subject"]


print("=" * 60)
print("RBF SVM HYPERPARAMETER TUNING")
print("=" * 60)

print("Sessions:", len(df))
print("Subjects:", df["subject"].nunique())
print("ROI features:", len(roi_columns))
print("Class 0:", (y == 0).sum())
print("Class 1:", (y == 1).sum())


# ============================================================
# PIPELINE
# ============================================================

pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),

    (
        "scaler",
        StandardScaler()
    ),

    (
        "pca",
        PCA(random_state=42)
    ),

    (
        "model",
        SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        )
    )
])


# ============================================================
# PARAMETER GRID
# ============================================================

param_grid = {

    "pca__n_components": [
        5,
        10,
        15,
        20,
        30
    ],

    "model__C": [
        0.1,
        0.5,
        1,
        2,
        5,
        10
    ],

    "model__gamma": [
        "scale",
        0.001,
        0.01,
        0.1
    ],

    "model__class_weight": [
        "balanced",
        None
    ]
}


# ============================================================
# GROUPED CV
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# GRID SEARCH
# ============================================================

print("\nStarting hyperparameter search...")

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True
)

search.fit(
    X,
    y,
    groups=groups
)


# ============================================================
# BEST PARAMETERS
# ============================================================

print("\n" + "=" * 60)
print("BEST PARAMETERS")
print("=" * 60)

print(search.best_params_)
print("\nBest CV ROC-AUC:", search.best_score_)


# ============================================================
# SAVE SEARCH RESULTS
# ============================================================

results = pd.DataFrame(
    search.cv_results_
)

results = results.sort_values(
    by="mean_test_score",
    ascending=False
)

output_file = (
    RESULTS_DIR
    / "svm_tuning_results.csv"
)

results.to_csv(
    output_file,
    index=False
)


print("\nTop 15 configurations:")

print(
    results[
        [
            "mean_test_score",
            "std_test_score",
            "param_pca__n_components",
            "param_model__C",
            "param_model__gamma",
            "param_model__class_weight"
        ]
    ]
    .head(15)
    .to_string(index=False)
)

print("\nSaved to:")
print(output_file)