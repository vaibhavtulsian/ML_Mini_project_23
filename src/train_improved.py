import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

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

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)
print("Unique subjects:", df["subject"].nunique())

print("\nTarget distribution:")
print(df["target"].value_counts().sort_index())


# ============================================================
# CREATE TWO FEATURE SETS
# ============================================================

roi_columns = [
    column
    for column in df.columns
    if column.startswith("ROI_")
]

print("\nNumber of ROI features:", len(roi_columns))


# fMRI only
X_fmri = df[roi_columns].copy()

# fMRI + demographics
X_demo = df[
    roi_columns + ["age", "sex"]
].copy()

y = df["target"].astype(int)

groups = df["subject"]


# ============================================================
# MODELS
# ============================================================

def create_models():

    return {

        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=42
                )
            )
        ]),

        "Linear SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    kernel="linear",
                    class_weight="balanced",
                    probability=True,
                    random_state=42
                )
            )
        ]),

        "RBF SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,
                    random_state=42
                )
            )
        ]),

        "Polynomial SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    kernel="poly",
                    degree=3,
                    class_weight="balanced",
                    probability=True,
                    random_state=42
                )
            )
        ]),

        "MLP": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    max_iter=1500,
                    early_stopping=True,
                    random_state=42
                )
            )
        ])
    }


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_models(X, feature_set_name):

    print("\n" + "=" * 60)
    print(feature_set_name)
    print("=" * 60)

    results = []

    models = create_models()

    for model_name, model in models.items():

        print("\n" + "-" * 50)
        print(model_name)
        print("-" * 50)

        fold_scores = []

        for fold, (train_idx, test_idx) in enumerate(
            cv.split(X, y, groups),
            start=1
        ):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            probabilities = model.predict_proba(
                X_test
            )[:, 1]

            balanced_accuracy = balanced_accuracy_score(
                y_test,
                predictions
            )

            precision = precision_score(
                y_test,
                predictions,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predictions,
                zero_division=0
            )

            try:
                roc_auc = roc_auc_score(
                    y_test,
                    probabilities
                )
            except ValueError:
                roc_auc = np.nan

            fold_scores.append({
                "balanced_accuracy": balanced_accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc
            })

            print(
                f"Fold {fold}: "
                f"BA={balanced_accuracy:.3f}, "
                f"Precision={precision:.3f}, "
                f"Recall={recall:.3f}, "
                f"F1={f1:.3f}, "
                f"AUC={roc_auc:.3f}"
            )

        results.append({

            "feature_set": feature_set_name,

            "model": model_name,

            "balanced_accuracy":
                np.nanmean([
                    x["balanced_accuracy"]
                    for x in fold_scores
                ]),

            "precision":
                np.nanmean([
                    x["precision"]
                    for x in fold_scores
                ]),

            "recall":
                np.nanmean([
                    x["recall"]
                    for x in fold_scores
                ]),

            "f1":
                np.nanmean([
                    x["f1"]
                    for x in fold_scores
                ]),

            "roc_auc":
                np.nanmean([
                    x["roc_auc"]
                    for x in fold_scores
                ])
        })

    return results


# ============================================================
# RUN BOTH EXPERIMENTS
# ============================================================

all_results = []

all_results.extend(
    evaluate_models(
        X_fmri,
        "fMRI Only"
    )
)

all_results.extend(
    evaluate_models(
        X_demo,
        "fMRI + Demographics"
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_df = results_df.sort_values(
    by="f1",
    ascending=False
)

output_file = (
    RESULTS_DIR
    / "improved_model_comparison.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print("\nSaved to:")
print(output_file)