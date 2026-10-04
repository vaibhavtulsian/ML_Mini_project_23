
# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import joblib
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier


# ============================================================
# 2. FILE PATHS
# ============================================================

DATA_FILE = "data/processed/ml_dataset.csv"

RESULTS_DIR = "results"

MODELS_DIR = "models"


# ============================================================
# 3. CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# PHASE 22
# LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("PHASE 22 - LOADING DATASET")
print("=" * 70)

print("\nLoading:")
print(DATA_FILE)


# Check whether CSV exists
if not os.path.exists(DATA_FILE):

    raise FileNotFoundError(
        f"\nCould not find:\n{DATA_FILE}\n\n"
        "Make sure ml_dataset.csv exists inside "
        "data/processed/."
    )


# Read CSV
df = pd.read_csv(DATA_FILE)


print("\nDataset loaded successfully.")

print("\nDataset shape:")
print(df.shape)


print("\nFirst 5 rows:")
print(df.head())


print("\nColumn names:")
print(df.columns.tolist())


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

if "subject" not in df.columns:

    raise ValueError(
        "\nERROR: 'subject' column is missing."
    )


if "target" not in df.columns:

    raise ValueError(
        "\nERROR: 'target' column is missing.\n\n"
        "Your CSV must contain an alcohol-use target column "
        "before training can begin."
    )


# ============================================================
# REMOVE ROWS WITH MISSING TARGET
# ============================================================

initial_rows = len(df)

df = df.dropna(subset=["target"])

removed_rows = initial_rows - len(df)


print("\nRows removed because of missing target:")

print(removed_rows)


# ============================================================
# CONVERT TARGET TO INTEGER
# ============================================================

try:

    df["target"] = df["target"].astype(int)

except ValueError:

    raise ValueError(
        "\nERROR: Target values must be numeric "
        "binary values such as 0 and 1."
    )


# ============================================================
# CHECK TARGET VALUES
# ============================================================

unique_targets = sorted(df["target"].unique())


print("\nUnique target values:")

print(unique_targets)


if not set(unique_targets).issubset({0, 1}):

    raise ValueError(
        "\nERROR: Target must contain only 0 and 1."
    )


if len(unique_targets) != 2:

    raise ValueError(
        "\nERROR: Both target classes must be present."
    )


# ============================================================
# PHASE 22
# SEPARATE FEATURES X AND TARGET y
# ============================================================

print("\n" + "=" * 70)
print("PHASE 22 - SEPARATING FEATURES AND TARGET")
print("=" * 70)


# X = input features
X = df.drop(
    columns=["subject", "target"]
)


# y = target
y = df["target"]


print("\nFeature matrix X shape:")

print(X.shape)


print("\nTarget vector y shape:")

print(y.shape)


# ============================================================
# HANDLE NON-NUMERIC FEATURES
# ============================================================

print("\nChecking feature data types...")

non_numeric_columns = X.select_dtypes(
    exclude=["number"]
).columns.tolist()


if len(non_numeric_columns) > 0:

    print(
        "\nNon-numeric columns found:"
    )

    print(non_numeric_columns)

    print(
        "\nConverting categorical features using "
        "one-hot encoding..."
    )

    X = pd.get_dummies(
        X,
        columns=non_numeric_columns,
        drop_first=True
    )


# ============================================================
# CONVERT EVERYTHING TO NUMERIC
# ============================================================

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# HANDLE MISSING FEATURE VALUES
# ============================================================

missing_values = X.isnull().sum().sum()


print("\nTotal missing feature values:")

print(missing_values)


if missing_values > 0:

    print(
        "\nReplacing missing feature values "
        "with column medians..."
    )

    X = X.fillna(
        X.median()
    )


# ============================================================
# FINAL FEATURE INFORMATION
# ============================================================

print("\nFinal feature matrix shape:")

print(X.shape)


print("\nNumber of features:")

print(X.shape[1])


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\nTarget distribution:")

print(
    y.value_counts()
    .sort_index()
)


print("\nTarget proportions:")

print(
    y.value_counts(
        normalize=True
    ).sort_index()
)


# ============================================================
# PHASE 23
# 10-FOLD STRATIFIED CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("PHASE 23 - 10-FOLD CROSS-VALIDATION")
print("=" * 70)


# Number of samples in smallest class
class_counts = y.value_counts()


minimum_class_count = class_counts.min()


print(
    "\nSmallest class contains:",
    minimum_class_count,
    "samples"
)


# 10-fold CV requires at least 10
# samples in the smallest class
if minimum_class_count < 10:

    raise ValueError(
        "\nERROR: There are not enough samples "
        "for 10-fold cross-validation.\n\n"
        f"Smallest class: {minimum_class_count} samples\n"
        "Required: at least 10 samples in the smallest class."
    )


# Create stratified CV
cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)


print(
    "\n10-fold StratifiedKFold created successfully."
)


# ============================================================
# PHASE 24
# FEATURE SCALING
# ============================================================

print("\n" + "=" * 70)
print("PHASE 24 - FEATURE SCALING")
print("=" * 70)


print(
    "\nStandardScaler will be placed inside "
    "each model pipeline."
)

print(
    "This prevents data leakage between "
    "training and validation folds."
)


# ============================================================
# PHASE 25
# LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("PHASE 25 - LOGISTIC REGRESSION")
print("=" * 70)


logistic_model = Pipeline(
    steps=[

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )

    ]
)


# ============================================================
# PHASE 26
# SUPPORT VECTOR MACHINES
# ============================================================

print("\n" + "=" * 70)
print("PHASE 26 - SUPPORT VECTOR MACHINES")
print("=" * 70)


# ------------------------------------------------------------
# Linear SVM
# ------------------------------------------------------------

linear_svm = Pipeline(
    steps=[

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            SVC(
                kernel="linear",
                probability=True,
                random_state=42
            )
        )

    ]
)


# ------------------------------------------------------------
# RBF SVM
# ------------------------------------------------------------

rbf_svm = Pipeline(
    steps=[

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            SVC(
                kernel="rbf",
                probability=True,
                random_state=42
            )
        )

    ]
)


# ------------------------------------------------------------
# Polynomial SVM
# ------------------------------------------------------------

polynomial_svm = Pipeline(
    steps=[

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            SVC(
                kernel="poly",
                probability=True,
                random_state=42
            )
        )

    ]
)


# ============================================================
# PHASE 27
# NEURAL NETWORK
# ============================================================

print("\n" + "=" * 70)
print("PHASE 27 - NEURAL NETWORK")
print("=" * 70)


mlp_model = Pipeline(
    steps=[

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            MLPClassifier(
                hidden_layer_sizes=(64, 32),

                max_iter=1000,

                early_stopping=True,

                random_state=42
            )
        )

    ]
)


# ============================================================
# STORE ALL MODELS
# ============================================================

models = {

    "Logistic Regression":
        logistic_model,

    "Linear SVM":
        linear_svm,

    "RBF SVM":
        rbf_svm,

    "Polynomial SVM":
        polynomial_svm,

    "Neural Network":
        mlp_model

}


# ============================================================
# PHASE 28
# DEFINE EVALUATION METRICS
# ============================================================

print("\n" + "=" * 70)
print("PHASE 28 - MODEL EVALUATION")
print("=" * 70)


scoring = {

    "accuracy":
        "accuracy",

    "precision":
        "precision",

    "recall":
        "recall",

    "f1":
        "f1",

    "roc_auc":
        "roc_auc"

}


# ============================================================
# RUN CROSS-VALIDATION
# ============================================================

results = []


for model_name, model in models.items():

    print("\n")
    print("-" * 70)

    print(
        "Training:",
        model_name
    )

    print("-" * 70)


    scores = cross_validate(

        estimator=model,

        X=X,

        y=y,

        cv=cv,

        scoring=scoring,

        n_jobs=-1

    )


    # --------------------------------------------------------
    # Calculate mean metrics
    # --------------------------------------------------------

    accuracy = (
        scores["test_accuracy"].mean()
    )

    precision = (
        scores["test_precision"].mean()
    )

    recall = (
        scores["test_recall"].mean()
    )

    f1 = (
        scores["test_f1"].mean()
    )

    roc_auc = (
        scores["test_roc_auc"].mean()
    )


    # --------------------------------------------------------
    # Calculate standard deviations
    # --------------------------------------------------------

    accuracy_std = (
        scores["test_accuracy"].std()
    )

    precision_std = (
        scores["test_precision"].std()
    )

    recall_std = (
        scores["test_recall"].std()
    )

    f1_std = (
        scores["test_f1"].std()
    )

    roc_auc_std = (
        scores["test_roc_auc"].std()
    )


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({

        "Model":
            model_name,

        "Accuracy":
            accuracy,

        "Accuracy Std":
            accuracy_std,

        "Precision":
            precision,

        "Precision Std":
            precision_std,

        "Recall":
            recall,

        "Recall Std":
            recall_std,

        "F1":
            f1,

        "F1 Std":
            f1_std,

        "ROC-AUC":
            roc_auc,

        "ROC-AUC Std":
            roc_auc_std

    })


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"Accuracy : {accuracy:.4f} "
        f"+/- {accuracy_std:.4f}"
    )

    print(
        f"Precision: {precision:.4f} "
        f"+/- {precision_std:.4f}"
    )

    print(
        f"Recall   : {recall:.4f} "
        f"+/- {recall_std:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f} "
        f"+/- {f1_std:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f} "
        f"+/- {roc_auc_std:.4f}"
    )


# ============================================================
# PHASE 29
# CREATE MODEL COMPARISON TABLE
# ============================================================

print("\n" + "=" * 70)
print("PHASE 29 - MODEL COMPARISON")
print("=" * 70)


results_df = pd.DataFrame(
    results
)


print(
    results_df.to_string(
        index=False,

        float_format=lambda x:
            f"{x:.4f}"
    )
)


# ============================================================
# PHASE 30
# SAVE RESULTS
# ============================================================

results_file = (
    f"{RESULTS_DIR}/model_comparison.csv"
)


results_df.to_csv(
    results_file,
    index=False
)


print("\n")
print(
    "Results saved to:"
)

print(
    results_file
)


# ============================================================
# PHASE 31
# SELECT MODEL
# ============================================================

print("\n" + "=" * 70)
print("PHASE 31 - MODEL SELECTION")
print("=" * 70)


# We use mean F1 score for model selection
best_index = (
    results_df["F1"].idxmax()
)


best_result = (
    results_df.loc[best_index]
)


best_model_name = (
    best_result["Model"]
)


print(
    "\nSelected model:",
    best_model_name
)


print(
    "\nPerformance:"
)


print(
    "Accuracy :",
    f"{best_result['Accuracy']:.4f}"
)


print(
    "Precision:",
    f"{best_result['Precision']:.4f}"
)


print(
    "Recall   :",
    f"{best_result['Recall']:.4f}"
)


print(
    "F1       :",
    f"{best_result['F1']:.4f}"
)


print(
    "ROC-AUC  :",
    f"{best_result['ROC-AUC']:.4f}"
)


# ============================================================
# PHASE 32
# TRAIN FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("PHASE 32 - TRAINING FINAL MODEL")
print("=" * 70)


# Retrieve selected model
final_model = models[
    best_model_name
]


print(
    "\nTraining",
    best_model_name,
    "on the complete dataset..."
)


final_model.fit(
    X,
    y
)


print(
    "Final model trained successfully."
)


# ============================================================
# PHASE 33
# SAVE FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("PHASE 33 - SAVING FINAL MODEL")
print("=" * 70)


# Create a safe filename
model_filename = (

    best_model_name
    .lower()
    .replace(" ", "_")
    .replace("-", "_")

    + ".pkl"

)


model_path = (
    f"{MODELS_DIR}/{model_filename}"
)


joblib.dump(
    final_model,
    model_path
)


print(
    "\nModel saved to:"
)

print(
    model_path
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)


print(
    "\nDataset:"
)

print(
    DATA_FILE
)


print(
    "\nNumber of subjects:",
    len(df)
)


print(
    "Number of features:",
    X.shape[1]
)


print(
    "\nSelected model:",
    best_model_name
)


print(
    "\nResults file:"
)

print(
    results_file
)


print(
    "\nSaved model:"
)

print(
    model_path
)


print(
    "\n" + "=" * 70
)