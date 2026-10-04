import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
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

X_fmri = df[roi_columns].copy()

X_demo = df[
    roi_columns + ["age", "sex"]
].copy()

y = df["target"].astype(int)
groups = df["subject"]


print("=" * 60)
print("PCA EXPERIMENT")
print("=" * 60)

print("Sessions:", len(df))
print("Subjects:", df["subject"].nunique())
print("ROI features:", len(roi_columns))
print("Positive:", (y == 1).sum())
print("Negative:", (y == 0).sum())


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=5000,
        C=1.0,
        random_state=42
    ),

    "Linear SVM": SVC(
        kernel="linear",
        class_weight="balanced",
        probability=True,
        C=1.0,
        random_state=42
    ),

    "RBF SVM": SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        C=1.0,
        gamma="scale",
        random_state=42
    )
}


# ============================================================
# PCA VALUES
# ============================================================

pca_values = [
    5,
    10,
    15,
    20,
    30
]


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


results = []


# ============================================================
# EXPERIMENT
# ============================================================

for feature_name, X in [
    ("fMRI Only", X_fmri),
    ("fMRI + Demographics", X_demo)
]:

    for n_components in pca_values:

        for model_name, classifier in models.items():

            scores = []

            print(
                f"\n{feature_name} | "
                f"PCA={n_components} | "
                f"{model_name}"
            )

            for fold, (train_idx, test_idx) in enumerate(
                cv.split(X, y, groups),
                start=1
            ):

                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]

                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

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
                        PCA(
                            n_components=n_components,
                            random_state=42
                        )
                    ),
                    (
                        "model",
                        classifier
                    )
                ])

                pipeline.fit(
                    X_train,
                    y_train
                )

                predictions = pipeline.predict(
                    X_test
                )

                probabilities = pipeline.predict_proba(
                    X_test
                )[:, 1]

                scores.append({
                    "balanced_accuracy":
                        balanced_accuracy_score(
                            y_test,
                            predictions
                        ),

                    "precision":
                        precision_score(
                            y_test,
                            predictions,
                            zero_division=0
                        ),

                    "recall":
                        recall_score(
                            y_test,
                            predictions,
                            zero_division=0
                        ),

                    "f1":
                        f1_score(
                            y_test,
                            predictions,
                            zero_division=0
                        ),

                    "roc_auc":
                        roc_auc_score(
                            y_test,
                            probabilities
                        )
                })

            results.append({

                "feature_set": feature_name,

                "pca_components": n_components,

                "model": model_name,

                "balanced_accuracy":
                    np.mean([
                        s["balanced_accuracy"]
                        for s in scores
                    ]),

                "precision":
                    np.mean([
                        s["precision"]
                        for s in scores
                    ]),

                "recall":
                    np.mean([
                        s["recall"]
                        for s in scores
                    ]),

                "f1":
                    np.mean([
                        s["f1"]
                        for s in scores
                    ]),

                "roc_auc":
                    np.mean([
                        s["roc_auc"]
                        for s in scores
                    ])
            })


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="roc_auc",
    ascending=False
)

output_file = (
    RESULTS_DIR
    / "pca_model_comparison.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("BEST PCA RESULTS")
print("=" * 70)

print(
    results_df.head(15).to_string(
        index=False
    )
)

print("\nSaved to:")
print(output_file)