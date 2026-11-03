
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------
# 1. PROJECT SETUP
# --------------------------------------------------


import os

RESULTS_DIR = r"C:\Users\hp\Documents\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_FILE = r"C:\Users\hp\Downloads\UNSW_NB15_training-set.csv"
TEST_FILE = r"C:\Users\hp\Downloads\UNSW_NB15_testing-set.csv"

print("=" * 60)
print("NETWORK INTRUSION DETECTION SYSTEM")
print("=" * 60)


# --------------------------------------------------
# 2. LOAD THE DATASET
# --------------------------------------------------

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("\nTraining dataset shape:", train_df.shape)
print("Testing dataset shape:", test_df.shape)

print("\nFirst five records:")
print(train_df.head())

print("\nDataset information:")
print(train_df.info())

# --------------------------------------------------
# 3. DATA CLEANING
# --------------------------------------------------

print("\nChecking missing values...")
print("Missing values in training data:",
      train_df.isnull().sum().sum())

print("Duplicate rows in training data:",
      train_df.duplicated().sum())

print("Missing values in testing data:",
      test_df.isnull().sum().sum())

# Replace infinite numeric values with missing values.
train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove duplicate rows if any are present.
train_df.drop_duplicates(inplace=True)
test_df.drop_duplicates(inplace=True)

# Remove records where the target label is missing.
train_df.dropna(subset=["label"], inplace=True)
test_df.dropna(subset=["label"], inplace=True)

print("\nData cleaning completed.")

# --------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# --------------------------------------------------

print("\nStatistical summary:")
print(train_df.describe().round(2))

# Save statistical summary.
train_df.describe(include="all").transpose().to_csv(
    os.path.join(RESULTS_DIR, "statistical_summary.csv")
)

# Plot 1: Normal versus attack traffic.
label_counts = train_df["label"].map(
    {0: "Normal", 1: "Attack"}
).fillna(train_df["label"].astype(str)).value_counts()

plt.figure(figsize=(7, 5))
label_counts.plot(kind="bar")
plt.title("Distribution of Normal and Attack Traffic")
plt.xlabel("Traffic Class")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "class_distribution.png")
)
plt.show()

# Plot 2: Distribution of attack categories.
if "attack_cat" in train_df.columns:
    attack_counts = train_df["attack_cat"].fillna(
        "Unknown"
    ).value_counts()

    plt.figure(figsize=(10, 6))
    attack_counts.plot(kind="bar")
    plt.title("Distribution of Network Traffic Categories")
    plt.xlabel("Traffic Category")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "attack_categories.png")
    )
    plt.show()

# Plot 3: Correlation heatmap for selected numeric features.
numeric_for_corr = train_df.select_dtypes(
    include=np.number
)

preferred_features = [
    col for col in [
        "dur", "spkts", "dpkts", "sbytes",
        "dbytes", "rate", "sttl", "dttl", "label"
    ]
    if col in numeric_for_corr.columns
]

if len(preferred_features) >= 2:
    corr = numeric_for_corr[preferred_features].corr()

    plt.figure(figsize=(10, 7))
    plt.imshow(corr, cmap="coolwarm", aspect="auto",
               vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(
        range(len(preferred_features)),
        preferred_features,
        rotation=45,
        ha="right"
    )
    plt.yticks(
        range(len(preferred_features)),
        preferred_features
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "correlation_heatmap.png")
    )
    plt.show()

# --------------------------------------------------
# 5. SELECT FEATURES AND TARGET
# --------------------------------------------------

# label = 0 means Normal; label = 1 means Attack.
# Exclude attack_cat to prevent target leakage.
# Exclude id because it is only an identifier.

X_train_full = train_df.drop(
    columns=["label", "attack_cat", "id"],
    errors="ignore"
)

y_train_full = train_df["label"].astype(int)

X_test_full = test_df.drop(
    columns=["label", "attack_cat", "id"],
    errors="ignore"
)

y_test_full = test_df["label"].astype(int)

# Keep only valid binary-class records.
train_mask = y_train_full.isin([0, 1])
test_mask = y_test_full.isin([0, 1])

X_train_full = X_train_full.loc[train_mask]
y_train_full = y_train_full.loc[train_mask]

X_test_full = X_test_full.loc[test_mask]
y_test_full = y_test_full.loc[test_mask]

# --------------------------------------------------
# 6. CREATE MANAGEABLE STRATIFIED SAMPLES
# --------------------------------------------------

print("\nPreparing samples for model training...")

X_train, _, y_train, _ = train_test_split(
    X_train_full,
    y_train_full,
    train_size=min(8000, len(X_train_full)),
    random_state=42,
    stratify=y_train_full
)

X_test, _, y_test, _ = train_test_split(
    X_test_full,
    y_test_full,
    train_size=min(3000, len(X_test_full)),
    random_state=42,
    stratify=y_test_full
)

print("Training sample:", X_train.shape)
print("Testing sample:", X_test.shape)

# --------------------------------------------------
# 7. DATA PREPROCESSING
# --------------------------------------------------

categorical_columns = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_columns = [
    col for col in X_train.columns
    if col not in categorical_columns
]

print("\nCategorical features:", categorical_columns)
print("Number of numerical features:", len(numerical_columns))

# Handle both newer and older scikit-learn versions.
try:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
except TypeError:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
    )

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", encoder)
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# --------------------------------------------------
# 8. DEFINE THE FIVE REQUIRED MODELS
# --------------------------------------------------

models = {
    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "SVM": LinearSVC(
        random_state=42,
        max_iter=5000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=12,
        class_weight="balanced",
        random_state=42
    ),

    "Naive Bayes": GaussianNB(),

    "Neural Network": MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=100,
        early_stopping=True,
        random_state=42
    )
}

# --------------------------------------------------
# 9. TRAIN AND EVALUATE ALL MODELS
# --------------------------------------------------

results = []
predictions = {}

for name, model in models.items():

    print("\n" + "=" * 50)
    print("Training model:", name)
    print("=" * 50)

    # Each model gets its own preprocessing pipeline.
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    start_time = time.time()

    pipeline.fit(X_train, y_train)

    training_time = time.time() - start_time

    y_pred = pipeline.predict(X_test)

    predictions[name] = y_pred

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Training Time (seconds)": training_time
    })

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("Training time:", round(training_time, 2), "seconds")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        zero_division=0
    ))

    report_text = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        zero_division=0
    )

    with open(
        os.path.join(
            RESULTS_DIR,
            name.replace(" ", "_") + "_report.txt"
        ),
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report_text)

# --------------------------------------------------
# 10. MODEL COMPARISON TABLE
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    os.path.join(RESULTS_DIR, "model_comparison.csv"),
    index=False
)

# --------------------------------------------------
# 11. MODEL COMPARISON CHART
# --------------------------------------------------

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

results_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(11, 6)
)

plt.title("Comparison of Five Classification Models")
plt.ylabel("Score")
plt.xlabel("Machine Learning Model")
plt.ylim(0, 1.05)
plt.xticks(rotation=20, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "model_comparison.png")
)
plt.show()

# --------------------------------------------------
# 12. CONFUSION MATRICES
# --------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, name in enumerate(models.keys()):

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions[name],
        labels=[0, 1],
        display_labels=["Normal", "Attack"],
        ax=axes[i],
        colorbar=False
    )

    axes[i].set_title(name)

# Hide the unused sixth subplot.
axes[-1].axis("off")

plt.suptitle("Confusion Matrices of All Five Models")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "confusion_matrices.png")
)
plt.show()

# --------------------------------------------------
# 13. FINAL SUMMARY
# --------------------------------------------------

best_model = results_df.iloc[0]

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Best model by F1 Score:", best_model["Model"])
print("F1 Score:", round(best_model["F1 Score"], 4))
print("Accuracy:", round(best_model["Accuracy"], 4))

print("\nAll charts, reports and comparison results are saved in:")
print(RESULTS_DIR)

print("\nRemember: these results use sampled records.")
print("They do not represent live network monitoring.")
