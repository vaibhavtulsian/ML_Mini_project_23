import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
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
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
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
print("FINAL MODEL EVALUATION")
print("=" * 60)

print("Sessions:", len(df))
print("Unique subjects:", df["subject"].nunique())
print("Features:", len(roi_columns))
print("Class 0:", (y == 0).sum())
print("Class 1:", (y == 1).sum())


# ============================================================
# FINAL MODEL
# ============================================================

model = Pipeline([
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
            n_components=5,
            random_state=42
        )
    ),

    (
        "model",
        SVC(
            kernel="rbf",
            C=5,
            gamma=0.001,
            class_weight="balanced",
            probability=True,
            random_state=42
        )
    )
])


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# STORAGE
# ============================================================

all_predictions = []
all_probabilities = []
all_actual = []
all_subjects = []

fold_results = []


# ============================================================
# RUN CV
# ============================================================

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups),
    start=1
):

    print("\n" + "=" * 60)
    print("FOLD", fold)
    print("=" * 60)

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

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    print(
        f"Balanced Accuracy: {balanced_accuracy:.4f}"
    )

    print(
        f"Precision:          {precision:.4f}"
    )

    print(
        f"Recall:             {recall:.4f}"
    )

    print(
        f"F1:                 {f1:.4f}"
    )

    print(
        f"ROC-AUC:            {auc:.4f}"
    )

    fold_results.append({
        "fold": fold,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc
    })

    all_predictions.extend(
        predictions
    )

    all_probabilities.extend(
        probabilities
    )

    all_actual.extend(
        y_test
    )

    all_subjects.extend(
        df.iloc[test_idx]["subject"]
    )


# ============================================================
# FOLD RESULTS
# ============================================================

fold_df = pd.DataFrame(
    fold_results
)

print("\n" + "=" * 60)
print("CROSS-VALIDATION RESULTS")
print("=" * 60)

print(
    fold_df.to_string(
        index=False
    )
)


print("\nMean ± Standard Deviation:")

for metric in [
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc"
]:

    mean = fold_df[metric].mean()
    std = fold_df[metric].std()

    print(
        f"{metric}: "
        f"{mean:.4f} ± {std:.4f}"
    )


# ============================================================
# SAVE FOLD RESULTS
# ============================================================

fold_df.to_csv(
    RESULTS_DIR / "final_fold_results.csv",
    index=False
)


# ============================================================
# COMBINED PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({
    "subject": all_subjects,
    "actual": all_actual,
    "prediction": all_predictions,
    "probability": all_probabilities
})

prediction_df.to_csv(
    RESULTS_DIR / "final_predictions.csv",
    index=False
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_actual,
    all_predictions
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Class 0", "Class 1"]
)

disp.plot()

plt.title(
    "Confusion Matrix - RBF SVM + PCA"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    all_actual,
    all_probabilities
)

overall_auc = roc_auc_score(
    all_actual,
    all_probabilities
)

plt.figure()

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {overall_auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve - RBF SVM + PCA"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "roc_curve.png",
    dpi=300
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL EVALUATION COMPLETE")
print("=" * 60)

print(
    "\nOverall ROC-AUC:",
    round(overall_auc, 4)
)

print("\nFiles created:")

print(
    RESULTS_DIR / "final_fold_results.csv"
)

print(
    RESULTS_DIR / "final_predictions.csv"
)

print(
    RESULTS_DIR / "confusion_matrix.png"
)

print(
    RESULTS_DIR / "roc_curve.png"
)