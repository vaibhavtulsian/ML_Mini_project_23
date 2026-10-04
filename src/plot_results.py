import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "data/processed/ml_dataset.csv"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

non_feature_columns = [
    "target",
    "subject",
    "session"
]

non_feature_columns = [
    col for col in non_feature_columns
    if col in df.columns
]

feature_columns = [
    col for col in df.columns
    if col not in non_feature_columns
]

X = df[feature_columns]
y = df["target"]

groups = df["subject"]


print("Dataset shape:", df.shape)
print("Features:", len(feature_columns))
print("Samples:", len(X))


# ============================================================
# 3. CROSS-VALIDATION
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


all_true = []
all_pred = []
all_prob = []


# ============================================================
# 4. RUN SAME TUNED PIPELINE
# ============================================================

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

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    all_true.extend(y_test)
    all_pred.extend(predictions)
    all_prob.extend(probabilities)


# Convert to arrays
all_true = np.array(all_true)
all_pred = np.array(all_pred)
all_prob = np.array(all_prob)


# ============================================================
# 5. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_true,
    all_pred
)

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print(cm)

print("\nTN:", cm[0, 0])
print("FP:", cm[0, 1])
print("FN:", cm[1, 0])
print("TP:", cm[1, 1])


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Class 0",
        "Class 1"
    ]
)

disp.plot()

plt.title(
    "Confusion Matrix - Tuned RBF SVM"
)

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 6. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    all_true,
    all_prob
)

auc = roc_auc_score(
    all_true,
    all_prob
)


print("\n========================================")
print("ROC-AUC")
print("========================================")

print(f"Overall ROC-AUC: {auc:.4f}")


plt.figure()

plt.plot(
    fpr,
    tpr,
    label=f"RBF SVM (AUC = {auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve - Tuned RBF SVM"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/roc_curve.png",
    dpi=300
)

plt.show()


# ============================================================
# 7. SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({
    "actual": all_true,
    "predicted": all_pred,
    "probability_class_1": all_prob
})

prediction_df.to_csv(
    "results/cv_predictions.csv",
    index=False
)


print("\n========================================")
print("FILES SAVED")
print("========================================")

print("results/confusion_matrix.png")
print("results/roc_curve.png")
print("results/cv_predictions.csv")