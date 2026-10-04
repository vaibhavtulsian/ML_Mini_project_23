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
    / "ml_dataset.csv"
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
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)
print("Unique subjects:", df["subject"].nunique())

print("\nTarget distribution:")
print(df["target"].value_counts().sort_index())


# ============================================================
# PREPARE X, Y AND GROUPS
# ============================================================

X = df.drop(
    columns=["subject", "session", "target"]
)

y = df["target"].astype(int)

groups = df["subject"]


# Convert everything to numeric
X = X.apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# MODELS
# ============================================================

models = {

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
                max_iter=1000,
                early_stopping=True,
                random_state=42
            )
        )
    ])
}


# ============================================================
# CROSS VALIDATION
# ============================================================

# Keep every subject entirely within one fold.
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

results = []

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

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

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(X_test)[:, 1]

        balanced_acc = balanced_accuracy_score(
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
            "balanced_accuracy": balanced_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc
        })

        print(
            f"Fold {fold}: "
            f"Balanced Acc={balanced_acc:.3f}, "
            f"Precision={precision:.3f}, "
            f"Recall={recall:.3f}, "
            f"F1={f1:.3f}, "
            f"ROC-AUC={roc_auc:.3f}"
        )

    # Average results across folds

    result = {
        "model": model_name,
        "balanced_accuracy_mean": np.nanmean(
            [x["balanced_accuracy"] for x in fold_scores]
        ),
        "precision_mean": np.nanmean(
            [x["precision"] for x in fold_scores]
        ),
        "recall_mean": np.nanmean(
            [x["recall"] for x in fold_scores]
        ),
        "f1_mean": np.nanmean(
            [x["f1"] for x in fold_scores]
        ),
        "roc_auc_mean": np.nanmean(
            [x["roc_auc"] for x in fold_scores]
        )
    }

    results.append(result)


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="f1_mean",
    ascending=False
)

output_file = (
    RESULTS_DIR
    / "model_comparison.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)

print("\nResults saved to:")
print(output_file)