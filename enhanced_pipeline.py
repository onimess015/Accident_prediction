"""
Enhanced Accident Severity Prediction Pipeline
Steps 4-7, 9-11: Feature Engineering, Multi-Model Training,
SHAP Explainability, Prediction Pipeline, Risk Map, Model Saving
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import xgboost as xgb
import shap
import folium
from folium.plugins import HeatMap
import joblib
import warnings

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")

# ============================================================
# STEP 1-3: LOAD & PREPARE DATA
# ============================================================
print("=" * 70)
print("STEP 1-3: LOADING & PREPARING DATASET")
print("=" * 70)

df = pd.read_csv("accident_prediction_india.csv")
print(f"Original shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Handle missing values
df["Traffic Control Presence"] = df["Traffic Control Presence"].fillna("Unknown")
df["Driver License Status"] = df["Driver License Status"].fillna("Unknown")

print(f"Missing values after fill: {df.isnull().sum().sum()}")

# ============================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: FEATURE ENGINEERING")
print("=" * 70)

# --- 1. rain_indicator ---
# WHY: Rain significantly reduces visibility and road grip, directly
# increasing accident severity. Binary flag isolates this effect.
df["rain_indicator"] = (df["Weather Conditions"] == "Rainy").astype(int)

# --- 2. night_indicator ---
# WHY: Nighttime driving has reduced visibility and higher fatigue risk.
# Captures the Dark/Dawn/Dusk lighting conditions as a single risk flag.
df["night_indicator"] = (
    df["Lighting Conditions"].isin(["Dark", "Dawn", "Dusk"]).astype(int)
)

# --- 3. weekend_indicator ---
# WHY: Weekend driving patterns differ (recreational travel, alcohol use).
# Captures behavioral risk factors tied to day of week.
df["weekend_indicator"] = df["Day of Week"].isin(["Saturday", "Sunday"]).astype(int)

# --- 4. traffic_density_score ---
# WHY: Higher vehicle counts and speed create more collision opportunities.
# Composite score combining vehicle count and speed exceeding limits.
# Normalized 0-1 for model compatibility.
speed_norm = df["Speed Limit (km/h)"] / df["Speed Limit (km/h)"].max()
vehicles_norm = (
    df["Number of Vehicles Involved"] / df["Number of Vehicles Involved"].max()
)
df["traffic_density_score"] = (0.5 * vehicles_norm + 0.5 * speed_norm).round(4)

# --- 5. road_risk_score ---
# WHY: Certain road type + condition combinations are inherently more dangerous.
# Encodes domain knowledge about road infrastructure risk levels.
road_type_risk = {
    "National Highway": 0.8,
    "State Highway": 0.7,
    "Urban Road": 0.5,
    "Rural Road": 0.6,
    "Expressway": 0.9,
}
road_cond_risk = {
    "Dry": 0.2,
    "Wet": 0.6,
    "Under Construction": 0.8,
    "Flooded": 1.0,
    "Icy": 0.9,
}
df["road_risk_score"] = (
    df["Road Type"].map(road_type_risk).fillna(0.5) * 0.5
    + df["Road Condition"].map(road_cond_risk).fillna(0.5) * 0.5
).round(4)

print("\nEngineered Features:")
print("  1. rain_indicator     - Binary: 1 if Rainy weather")
print("     WHY: Rain reduces visibility & grip -> higher severity")
print(f"     Distribution: {df['rain_indicator'].value_counts().to_dict()}")

print("  2. night_indicator    - Binary: 1 if Dark/Dawn/Dusk")
print("     WHY: Low visibility + driver fatigue -> more severe accidents")
print(f"     Distribution: {df['night_indicator'].value_counts().to_dict()}")

print("  3. weekend_indicator  - Binary: 1 if Saturday/Sunday")
print("     WHY: Different driving patterns, higher alcohol involvement")
print(f"     Distribution: {df['weekend_indicator'].value_counts().to_dict()}")

print("  4. traffic_density_score - Composite: vehicles + speed (0-1)")
print("     WHY: More vehicles + higher speed -> greater collision risk")
print(
    f"     Stats: mean={df['traffic_density_score'].mean():.3f}, std={df['traffic_density_score'].std():.3f}"
)

print("  5. road_risk_score    - Composite: road type + condition (0-1)")
print("     WHY: Infrastructure quality directly affects crash outcomes")
print(
    f"     Stats: mean={df['road_risk_score'].mean():.3f}, std={df['road_risk_score'].std():.3f}"
)

# ============================================================
# STEP 5: TRAIN MACHINE LEARNING MODELS
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: TRAINING MACHINE LEARNING MODELS")
print("=" * 70)

# Define features and target
feature_cols = [
    "Vehicle Type Involved",
    "Weather Conditions",
    "Road Type",
    "Lighting Conditions",
    "Road Condition",
    "Speed Limit (km/h)",
    "Alcohol Involvement",
    "Driver Age",
    "Driver Gender",
    "Traffic Control Presence",
    "Number of Vehicles Involved",
    "rain_indicator",
    "night_indicator",
    "weekend_indicator",
    "traffic_density_score",
    "road_risk_score",
]

cat_features = [
    "Vehicle Type Involved",
    "Weather Conditions",
    "Road Type",
    "Lighting Conditions",
    "Road Condition",
    "Alcohol Involvement",
    "Driver Gender",
    "Traffic Control Presence",
]
num_features = [
    "Speed Limit (km/h)",
    "Driver Age",
    "Number of Vehicles Involved",
    "rain_indicator",
    "night_indicator",
    "weekend_indicator",
    "traffic_density_score",
    "road_risk_score",
]

X = df[feature_cols]
y = df["Accident Severity"]

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Fatal=0, Minor=1, Serious=2

# Preprocessor
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

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")
print(f"Classes: {list(le.classes_)}")

# Transform features
X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()
print(f"Total features after encoding: {len(feature_names)}")

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    ),
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    model.fit(X_train_enc, y_train)
    y_pred = model.predict(X_test_enc)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1}
    trained_models[name] = model

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print(cm_df.to_string().replace("\n", "\n  "))

    print(f"\n  Classification Report:")
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    print("  " + report.replace("\n", "\n  "))

# --- Model Comparison Table ---
print("\n" + "=" * 70)
print("MODEL COMPARISON TABLE")
print("=" * 70)
comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df.sort_values("F1 Score", ascending=False)
print(f"\n{comparison_df.to_string()}")

best_model_name = comparison_df.index[0]
best_model = trained_models[best_model_name]
print(f"\nBest model: {best_model_name} (F1={comparison_df.iloc[0]['F1 Score']:.4f})")

# Save comparison chart
fig, ax = plt.subplots(figsize=(12, 6))
comparison_df.plot(kind="bar", ax=ax, colormap="viridis")
ax.set_title("Model Comparison: All Metrics", fontsize=14, fontweight="bold")
ax.set_ylabel("Score")
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
ax.legend(loc="lower right")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: model_comparison.png")

# ============================================================
# STEP 6: FEATURE IMPORTANCE & EXPLAINABILITY
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: FEATURE IMPORTANCE & EXPLAINABILITY")
print("=" * 70)

# --- 6a. Model Feature Importance (Random Forest) ---
rf_model = trained_models["Random Forest"]
rf_importances = pd.Series(rf_model.feature_importances_, index=feature_names)
rf_importances = rf_importances.sort_values(ascending=False)

print("\n--- Random Forest Feature Importance (Top 15) ---")
top15_rf = rf_importances.head(15)
for feat, imp in top15_rf.items():
    print(f"  {feat}: {imp:.1%}")

# Plot RF importance
fig, ax = plt.subplots(figsize=(12, 8))
top15_plot = top15_rf.sort_values(ascending=True)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top15_plot)))
ax.barh(top15_plot.index, top15_plot.values, color=colors)
ax.set_title(
    "Top 15 Feature Importances (Random Forest)", fontsize=14, fontweight="bold"
)
ax.set_xlabel("Importance")
for i, v in enumerate(top15_plot.values):
    ax.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: feature_importance.png")

# --- 6b. SHAP Values (using best tree-based model) ---
print("\n--- Computing SHAP Values ---")
# Use a small sample for SHAP (faster computation)
shap_sample_size = min(100, X_test_enc.shape[0])
X_shap = X_test_enc[:shap_sample_size]

# Use XGBoost for SHAP (much faster than RF)
xgb_model = trained_models["XGBoost"]
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_shap)

# SHAP summary plot
fig, ax = plt.subplots(figsize=(14, 10))
# For multi-class, average absolute SHAP values across classes
if isinstance(shap_values, list):
    mean_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
else:
    mean_shap = (
        np.abs(shap_values).mean(axis=(0, 2))
        if shap_values.ndim == 3
        else np.abs(shap_values).mean(axis=0)
    )

shap_importance = pd.Series(mean_shap.mean(axis=0), index=feature_names).sort_values(
    ascending=False
)

top15_shap = shap_importance.head(15).sort_values(ascending=True)
colors_shap = plt.cm.magma(np.linspace(0.3, 0.85, len(top15_shap)))
ax.barh(top15_shap.index, top15_shap.values, color=colors_shap)
ax.set_title("Top 15 Features by SHAP Importance", fontsize=14, fontweight="bold")
ax.set_xlabel("Mean |SHAP Value|")
for i, v in enumerate(top15_shap.values):
    ax.text(v + 0.0005, i, f"{v:.4f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("shap_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_importance.png")

# Print top factors with percentages
print("\n--- Top Factors Influencing Accidents ---")
total_shap = shap_importance.sum()
for feat, val in shap_importance.head(10).items():
    pct = val / total_shap * 100
    print(f"  {feat}: {pct:.1f}%")

# SHAP bar plot per class
try:
    fig = plt.figure(figsize=(14, 8))
    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=list(feature_names),
        plot_type="bar",
        show=False,
        max_display=15,
    )
    plt.title("SHAP Feature Importance by Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("shap_bar_by_class.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: shap_bar_by_class.png")
except Exception as e:
    print(f"  SHAP class plot skipped: {e}")

# ============================================================
# STEP 7: PREDICTION PIPELINE
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: PREDICTION PIPELINE")
print("=" * 70)


def predict_accident_risk(
    weather,
    road_type,
    road_condition,
    lighting,
    vehicle_type,
    speed,
    alcohol,
    driver_age,
    driver_gender,
    traffic_control,
    num_vehicles,
    day_of_week,
    model=best_model,
    preprocessor=preprocessor,
    label_encoder=le,
):
    """
    Reusable prediction function.

    Input: accident scenario parameters
    Output: dict with probability, predicted class, and risk level
    """
    # Build engineered features
    rain = 1 if weather == "Rainy" else 0
    night = 1 if lighting in ["Dark", "Dawn", "Dusk"] else 0
    weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0

    speed_n = speed / 120  # max speed in dataset
    vehicles_n = num_vehicles / 10  # max vehicles
    traffic_density = round(0.5 * vehicles_n + 0.5 * speed_n, 4)

    rt_risk = road_type_risk.get(road_type, 0.5)
    rc_risk = road_cond_risk.get(road_condition, 0.5)
    road_risk = round(rt_risk * 0.5 + rc_risk * 0.5, 4)

    input_data = pd.DataFrame(
        [
            {
                "Vehicle Type Involved": vehicle_type,
                "Weather Conditions": weather,
                "Road Type": road_type,
                "Lighting Conditions": lighting,
                "Road Condition": road_condition,
                "Speed Limit (km/h)": speed,
                "Alcohol Involvement": alcohol,
                "Driver Age": driver_age,
                "Driver Gender": driver_gender,
                "Traffic Control Presence": traffic_control,
                "Number of Vehicles Involved": num_vehicles,
                "rain_indicator": rain,
                "night_indicator": night,
                "weekend_indicator": weekend,
                "traffic_density_score": traffic_density,
                "road_risk_score": road_risk,
            }
        ]
    )

    X_enc = preprocessor.transform(input_data)
    probabilities = model.predict_proba(X_enc)[0]
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = label_encoder.classes_[predicted_class_idx]
    max_prob = probabilities[predicted_class_idx]

    # Risk level based on Fatal probability
    fatal_idx = list(label_encoder.classes_).index("Fatal")
    fatal_prob = probabilities[fatal_idx]

    if fatal_prob >= 0.5:
        risk_level = "HIGH"
    elif fatal_prob >= 0.3:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    return {
        "predicted_severity": predicted_class,
        "confidence": round(max_prob, 4),
        "probabilities": {
            label_encoder.classes_[i]: round(p, 4) for i, p in enumerate(probabilities)
        },
        "accident_risk": round(fatal_prob, 4),
        "risk_level": risk_level,
    }


# Test the prediction function
print("\n--- Example Prediction 1: High Risk Scenario ---")
result1 = predict_accident_risk(
    weather="Rainy",
    road_type="National Highway",
    road_condition="Wet",
    lighting="Dark",
    vehicle_type="Two-Wheeler",
    speed=90,
    alcohol="Yes",
    driver_age=22,
    driver_gender="Male",
    traffic_control="None",
    num_vehicles=3,
    day_of_week="Saturday",
)
print(f"  Predicted Severity: {result1['predicted_severity']}")
print(f"  Confidence: {result1['confidence']:.2%}")
print(f"  Probabilities: {result1['probabilities']}")
print(f"  Accident Risk: {result1['accident_risk']:.2f}")
print(f"  Risk Level: {result1['risk_level']}")

print("\n--- Example Prediction 2: Low Risk Scenario ---")
result2 = predict_accident_risk(
    weather="Clear",
    road_type="Urban Road",
    road_condition="Dry",
    lighting="Daylight",
    vehicle_type="Car",
    speed=40,
    alcohol="No",
    driver_age=35,
    driver_gender="Female",
    traffic_control="Signals",
    num_vehicles=1,
    day_of_week="Wednesday",
)
print(f"  Predicted Severity: {result2['predicted_severity']}")
print(f"  Confidence: {result2['confidence']:.2%}")
print(f"  Probabilities: {result2['probabilities']}")
print(f"  Accident Risk: {result2['accident_risk']:.2f}")
print(f"  Risk Level: {result2['risk_level']}")

# ============================================================
# STEP 9: ACCIDENT RISK MAP
# ============================================================
print("\n" + "=" * 70)
print("STEP 9: ACCIDENT RISK MAP")
print("=" * 70)

# State coordinates for India
state_coords = {
    "Andhra Pradesh": [15.9129, 79.7400],
    "Arunachal Pradesh": [28.2180, 94.7278],
    "Assam": [26.2006, 92.9376],
    "Bihar": [25.0961, 85.3131],
    "Chhattisgarh": [21.2787, 81.8661],
    "Goa": [15.2993, 74.1240],
    "Gujarat": [22.2587, 71.1924],
    "Haryana": [29.0588, 76.0856],
    "Himachal Pradesh": [31.1048, 77.1734],
    "Jharkhand": [23.6102, 85.2799],
    "Karnataka": [15.3173, 75.7139],
    "Kerala": [10.8505, 76.2711],
    "Madhya Pradesh": [22.9734, 78.6569],
    "Maharashtra": [19.7515, 75.7139],
    "Manipur": [24.6637, 93.9063],
    "Meghalaya": [25.4670, 91.3662],
    "Mizoram": [23.1645, 92.9376],
    "Nagaland": [26.1584, 94.5624],
    "Odisha": [20.9517, 85.0985],
    "Punjab": [31.1471, 75.3412],
    "Rajasthan": [27.0238, 74.2179],
    "Sikkim": [27.5330, 88.5122],
    "Tamil Nadu": [11.1271, 78.6569],
    "Telangana": [18.1124, 79.0193],
    "Tripura": [23.9408, 91.9882],
    "Uttar Pradesh": [26.8467, 80.9462],
    "Uttarakhand": [30.0668, 79.0193],
    "West Bengal": [22.9868, 87.8550],
    "Delhi": [28.7041, 77.1025],
    "Jammu and Kashmir": [33.7782, 76.5762],
    "Chandigarh": [30.7333, 76.7794],
    "Puducherry": [11.9416, 79.8083],
}

# Compute state-level risk
severity_map = {"Minor": 1, "Serious": 2, "Fatal": 3}
df["Severity_Score"] = df["Accident Severity"].map(severity_map)

state_risk = (
    df.groupby("State Name")
    .agg(
        avg_severity=("Severity_Score", "mean"),
        total_accidents=("Severity_Score", "count"),
        fatal_count=("Accident Severity", lambda x: (x == "Fatal").sum()),
    )
    .reset_index()
)
state_risk["fatal_rate"] = state_risk["fatal_count"] / state_risk["total_accidents"]

# Create Folium map
india_map = folium.Map(
    location=[20.5937, 78.9629], zoom_start=5, tiles="cartodbpositron"
)

# Add heatmap data
heat_data = []
for _, row in state_risk.iterrows():
    coords = state_coords.get(row["State Name"])
    if coords:
        # Weight by severity and accident count
        weight = (
            row["avg_severity"]
            * row["total_accidents"]
            / state_risk["total_accidents"].max()
        )
        heat_data.append([coords[0], coords[1], weight])

HeatMap(heat_data, radius=30, blur=20, max_zoom=6).add_to(india_map)

# Add color-coded markers
for _, row in state_risk.iterrows():
    coords = state_coords.get(row["State Name"])
    if coords:
        sev = row["avg_severity"]
        if sev >= 2.2:
            color, level = "red", "HIGH RISK"
        elif sev >= 1.9:
            color, level = "orange", "MODERATE"
        else:
            color, level = "green", "LOW RISK"

        popup_text = (
            f"<b>{row['State Name']}</b><br>"
            f"Accidents: {row['total_accidents']}<br>"
            f"Fatal: {row['fatal_count']} ({row['fatal_rate']:.0%})<br>"
            f"Avg Severity: {sev:.2f}/3<br>"
            f"Risk: <b style='color:{color}'>{level}</b>"
        )
        folium.CircleMarker(
            location=coords,
            radius=max(5, row["total_accidents"] / 8),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=250),
            tooltip=f"{row['State Name']}: {level}",
        ).add_to(india_map)

# Add legend
legend_html = """
<div style="position:fixed;bottom:50px;left:50px;z-index:1000;
background:white;padding:10px;border-radius:5px;border:2px solid grey;font-size:14px;">
<b>Accident Risk Level</b><br>
<i style="background:red;width:12px;height:12px;display:inline-block;border-radius:50%;"></i> High Risk (Severity ≥ 2.2)<br>
<i style="background:orange;width:12px;height:12px;display:inline-block;border-radius:50%;"></i> Moderate (1.9 - 2.2)<br>
<i style="background:green;width:12px;height:12px;display:inline-block;border-radius:50%;"></i> Low Risk (< 1.9)
</div>
"""
india_map.get_root().html.add_child(folium.Element(legend_html))

india_map.save("accident_risk_map.html")
print("Saved: accident_risk_map.html")
print(
    f"States mapped: {len([s for s in state_risk['State Name'] if s in state_coords])}"
)

# Print risk zone summary
print("\n--- Risk Zones ---")
for _, row in state_risk.sort_values("avg_severity", ascending=False).iterrows():
    sev = row["avg_severity"]
    if sev >= 2.2:
        zone = "RED (HIGH)"
    elif sev >= 1.9:
        zone = "YELLOW (MODERATE)"
    else:
        zone = "GREEN (LOW)"
    print(
        f"  {row['State Name']:25s} | Severity: {sev:.2f} | Accidents: {row['total_accidents']:3d} | Zone: {zone}"
    )

# ============================================================
# STEP 11: MODEL SAVING
# ============================================================
print("\n" + "=" * 70)
print("STEP 11: SAVING MODELS & ARTIFACTS")
print("=" * 70)

# Save best model + all components needed for prediction
artifacts = {
    "best_model": best_model,
    "best_model_name": best_model_name,
    "preprocessor": preprocessor,
    "label_encoder": le,
    "feature_cols": feature_cols,
    "cat_features": cat_features,
    "num_features": num_features,
    "road_type_risk": road_type_risk,
    "road_cond_risk": road_cond_risk,
    "results": results,
}

joblib.dump(artifacts, "model.pkl")
print("Saved: model.pkl (contains model + preprocessor + encoders)")

# Also save individual best model
joblib.dump(best_model, "accident_severity_model.pkl")
print("Saved: accident_severity_model.pkl (best model only)")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("PIPELINE COMPLETE — FINAL SUMMARY")
print("=" * 70)

print(f"\n  Best Model: {best_model_name}")
print(f"  Accuracy:   {results[best_model_name]['Accuracy']:.4f}")
print(f"  F1 Score:   {results[best_model_name]['F1 Score']:.4f}")

print(f"\n  Engineered Features: rain_indicator, night_indicator, weekend_indicator,")
print(f"                       traffic_density_score, road_risk_score")

print(f"\n  Output Files:")
print(f"    - model.pkl                  (full pipeline artifact)")
print(f"    - accident_severity_model.pkl (best model)")
print(f"    - model_comparison.png       (metric comparison chart)")
print(f"    - feature_importance.png     (RF feature importance)")
print(f"    - shap_importance.png        (SHAP feature importance)")
print(f"    - shap_bar_by_class.png      (SHAP by severity class)")
print(f"    - accident_risk_map.html     (interactive risk map)")

print(f"\n{'=' * 70}")
