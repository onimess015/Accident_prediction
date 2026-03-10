import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================
print("=" * 70)
print("STEP 1: LOADING DATASET")
print("=" * 70)

df = pd.read_csv("accident_prediction_india.csv")

print(f"\nShape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nDataset Info:")
df.info()

# ============================================================
# STEP 2: SELECT REQUIRED COLUMNS
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: SELECTING REQUIRED COLUMNS")
print("=" * 70)

required_columns = [
    "Vehicle Type Involved",
    "Weather Conditions",
    "Road Type",
    "Lighting Conditions",
    "Road Condition",
    "Speed Limit (km/h)",
    "Alcohol Involvement",
    "Accident Severity",
]

df = df[required_columns]
print(f"\nColumns kept: {list(df.columns)}")
print(f"Shape after selection: {df.shape}")

# ============================================================
# STEP 3: HANDLE MISSING DATA
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: HANDLING MISSING DATA")
print("=" * 70)

print(f"\nMissing values before:\n{df.isnull().sum()}")

categorical_cols = df.select_dtypes(include="object").columns.tolist()
numeric_cols = df.select_dtypes(include="number").columns.tolist()

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna("Unknown")

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

print(f"\nMissing values after:\n{df.isnull().sum()}")

# ============================================================
# STEP 4: DEFINE TARGET AND FEATURES
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: DEFINING TARGET AND FEATURES")
print("=" * 70)

y = df["Accident Severity"]
X = df.drop(columns=["Accident Severity"])

print(f"\nTarget (y): Accident Severity — {y.nunique()} classes: {y.unique().tolist()}")
print(f"Features (X): {list(X.columns)}")
print(f"X shape: {X.shape}")

# ============================================================
# STEP 5: ENCODE CATEGORICAL VARIABLES
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: ENCODING CATEGORICAL VARIABLES")
print("=" * 70)

cat_features = [
    "Vehicle Type Involved",
    "Weather Conditions",
    "Road Type",
    "Lighting Conditions",
    "Road Condition",
    "Alcohol Involvement",
]
num_features = ["Speed Limit (km/h)"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_features,
        ),
        ("num", "passthrough", num_features),
    ]
)

X_transformed = preprocessor.fit_transform(X)
feature_names = preprocessor.get_feature_names_out()

print(f"\nEncoded feature count: {X_transformed.shape[1]}")
print(f"Sample feature names: {list(feature_names[:10])}...")

# ============================================================
# STEP 6: TRAIN TEST SPLIT
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: TRAIN TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train distribution:\n{y_train.value_counts()}")
print(f"y_test distribution:\n{y_test.value_counts()}")

# ============================================================
# STEP 7: BUILD MACHINE LEARNING MODEL
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: TRAINING RANDOM FOREST MODEL")
print("=" * 70)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
)

model.fit(X_train, y_train)
print("\nModel training completed successfully.")

# ============================================================
# STEP 8: MODEL EVALUATION
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: MODEL EVALUATION")
print("=" * 70)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy Score: {acc:.4f} ({acc:.2%})")

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print(pd.DataFrame(cm, index=model.classes_, columns=model.classes_))

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# ============================================================
# STEP 9: FEATURE IMPORTANCE VISUALIZATION
# ============================================================
print("\n" + "=" * 70)
print("STEP 9: FEATURE IMPORTANCE VISUALIZATION")
print("=" * 70)

importances = model.feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

top15 = feat_imp.head(15).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top15)))
ax.barh(top15.index, top15.values, color=colors)
ax.set_title(
    "Top 15 Feature Importances (Random Forest)", fontsize=14, fontweight="bold"
)
ax.set_xlabel("Importance")
for i, v in enumerate(top15.values):
    ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

print("Feature importance chart saved as: feature_importance.png")

# ============================================================
# STEP 10: SAVE TRAINED MODEL
# ============================================================
print("\n" + "=" * 70)
print("STEP 10: SAVING TRAINED MODEL")
print("=" * 70)

joblib.dump(model, "accident_severity_model.pkl")
print("Model saved as: accident_severity_model.pkl")

# ============================================================
# STEP 11: FINAL CONSOLE SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"\n  Model Training: COMPLETED")
print(f"  Accuracy Score: {acc:.4f} ({acc:.2%})")

print(f"\n  Top 5 Most Important Features:")
for i, (feat, imp) in enumerate(feat_imp.head(5).items(), 1):
    print(f"    {i}. {feat} — {imp:.4f}")

print(f"\n  Saved Files:")
print(f"    - accident_severity_model.pkl  (trained model)")
print(f"    - feature_importance.png       (feature importance chart)")
print(f"\n{'=' * 70}")
print("PIPELINE COMPLETE")
print("=" * 70)
