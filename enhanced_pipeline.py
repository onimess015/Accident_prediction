"""
MAXIMUM ACCURACY Accident Severity Prediction Pipeline
Uses ALL available features + aggressive feature engineering + ensemble stacking
Includes complete India state + city coordinates for risk map
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
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
# COMPLETE INDIA COORDINATES DATABASE
# ============================================================

# All 28 States + 8 Union Territories
STATE_COORDS = {
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
    "Ladakh": [34.1526, 77.5771],
    "Lakshadweep": [10.5667, 72.6417],
    "Andaman and Nicobar Islands": [11.7401, 92.6586],
    "Dadra and Nagar Haveli and Daman and Diu": [20.1809, 73.0169],
}

# All cities in dataset + 100 major Indian cities
CITY_COORDS = {
    "Lucknow": [26.8467, 80.9462],
    "Jodhpur": [26.2389, 73.0243],
    "Kanpur": [26.4499, 80.3319],
    "Tirupati": [13.6288, 79.4192],
    "Bangalore": [12.9716, 77.5946],
    "Varanasi": [25.3176, 82.9739],
    "Nagpur": [21.1458, 79.0882],
    "Madurai": [9.9252, 78.1198],
    "Surat": [21.1702, 72.8311],
    "Mumbai": [19.0760, 72.8777],
    "Durgapur": [23.5204, 87.3119],
    "Kolkata": [22.5726, 88.3639],
    "Ahmedabad": [23.0225, 72.5714],
    "Mangalore": [12.9141, 74.8560],
    "Siliguri": [26.7271, 88.3953],
    "Vijayawada": [16.5062, 80.6480],
    "Vadodara": [22.3072, 73.1812],
    "Mysore": [12.2958, 76.6394],
    "Dwarka": [22.2394, 68.9678],
    "Chennai": [13.0827, 80.2707],
    "Rohini": [28.7495, 77.0565],
    "Visakhapatnam": [17.6868, 83.2185],
    "New Delhi": [28.6139, 77.2090],
    "Udaipur": [24.5854, 73.7125],
    "Coimbatore": [11.0168, 76.9558],
    "Pune": [18.5204, 73.8567],
    "Jaipur": [26.9124, 75.7873],
    "Hyderabad": [17.3850, 78.4867],
    "Delhi": [28.7041, 77.1025],
    "Patna": [25.6093, 85.1376],
    "Bhopal": [23.2599, 77.4126],
    "Bhubaneswar": [20.2961, 85.8245],
    "Dehradun": [30.3165, 78.0322],
    "Ranchi": [23.3441, 85.3096],
    "Raipur": [21.2514, 81.6296],
    "Shimla": [31.1048, 77.1734],
    "Gangtok": [27.3389, 88.6065],
    "Shillong": [25.5788, 91.8933],
    "Aizawl": [23.7271, 92.7176],
    "Kohima": [25.6751, 94.1086],
    "Imphal": [24.8170, 93.9368],
    "Agartala": [23.8315, 91.2868],
    "Itanagar": [27.0844, 93.6053],
    "Guwahati": [26.1445, 91.7362],
    "Thiruvananthapuram": [8.5241, 76.9366],
    "Panaji": [15.4909, 73.8278],
    "Amritsar": [31.6340, 74.8723],
    "Chandigarh": [30.7333, 76.7794],
    "Indore": [22.7196, 75.8577],
    "Thane": [19.2183, 72.9781],
    "Kochi": [9.9312, 76.2673],
    "Agra": [27.1767, 78.0081],
    "Meerut": [28.9845, 77.7064],
    "Nashik": [19.9975, 73.7898],
    "Jabalpur": [23.1815, 79.9864],
    "Allahabad": [25.4358, 81.8463],
    "Gwalior": [26.2183, 78.1828],
    "Aurangabad": [19.8762, 75.3433],
    "Rajkot": [22.3039, 70.8022],
    "Dhanbad": [23.7957, 86.4304],
    "Jamshedpur": [22.8046, 86.2029],
    "Bokaro": [23.6693, 86.1511],
    "Asansol": [23.6739, 86.9524],
    "Navi Mumbai": [19.0330, 73.0297],
    "Faridabad": [28.4089, 77.3178],
    "Ghaziabad": [28.6692, 77.4538],
    "Noida": [28.5355, 77.3910],
    "Howrah": [22.5958, 88.2636],
    "Cuttack": [20.4625, 85.8830],
    "Warangal": [17.9784, 79.5941],
    "Guntur": [16.3067, 80.4365],
    "Nellore": [14.4426, 79.9865],
    "Rajahmundry": [17.0005, 81.8040],
    "Kakinada": [16.9891, 82.2475],
    "Tiruchirappalli": [10.7905, 78.7047],
    "Salem": [11.6643, 78.1460],
    "Erode": [11.3410, 77.7172],
    "Vellore": [12.9165, 79.1325],
    "Hubli": [15.3647, 75.1240],
    "Belgaum": [15.8497, 74.4977],
    "Gulbarga": [17.3297, 76.8343],
    "Davangere": [14.4644, 75.9218],
    "Bellary": [15.1394, 76.9214],
    "Bikaner": [28.0229, 73.3119],
    "Ajmer": [26.4499, 74.6399],
    "Kota": [25.2138, 75.8648],
    "Bhilwara": [25.3407, 74.6313],
    "Alwar": [27.5530, 76.6346],
    "Sikar": [27.6094, 75.1399],
    "Pali": [25.7711, 73.3234],
    "Bharatpur": [27.2152, 77.5030],
    "Gorakhpur": [26.7606, 83.3732],
    "Moradabad": [28.8386, 78.7733],
    "Aligarh": [27.8974, 78.0880],
    "Bareilly": [28.3670, 79.4304],
    "Saharanpur": [29.9680, 77.5510],
    "Jhansi": [25.4484, 78.5685],
    "Mathura": [27.4924, 77.6737],
    "Firozabad": [27.1591, 78.3957],
    "Muzaffarnagar": [29.4727, 77.7085],
    "Rae Bareli": [26.2345, 81.2329],
    "Sultanpur": [26.2648, 82.0727],
    "Ludhiana": [30.9010, 75.8573],
    "Jalandhar": [31.3260, 75.5762],
    "Patiala": [30.3398, 76.3869],
    "Bathinda": [30.2110, 74.9455],
    "Jammu": [32.7266, 74.8570],
    "Srinagar": [34.0837, 74.7973],
    "Haridwar": [29.9457, 78.1642],
    "Rishikesh": [30.0869, 78.2676],
    "Nainital": [29.3803, 79.4636],
    "Dharamsala": [32.2190, 76.3234],
    "Manali": [32.2396, 77.1887],
    "Panipat": [29.3909, 76.9635],
    "Karnal": [29.6857, 76.9905],
    "Rohtak": [28.8955, 76.6066],
    "Hisar": [29.1492, 75.7217],
    "Sonipat": [28.9931, 77.0151],
    "Gurugram": [28.4595, 77.0266],
    "Jodhpur": [26.2389, 73.0243],
    "Ujjain": [23.1765, 75.7885],
    "Rewa": [24.5373, 81.3042],
    "Satna": [24.5682, 80.8322],
    "Sagar": [23.8388, 78.7378],
    "Bilaspur": [22.0796, 82.1391],
    "Korba": [22.3595, 82.7501],
    "Durg": [21.1904, 81.2849],
    "Bhilai": [21.2093, 81.3780],
    "Sambalpur": [21.4669, 83.9812],
    "Berhampur": [19.3150, 84.7941],
    "Rourkela": [22.2604, 84.8536],
    "Silchar": [24.8333, 92.7789],
    "Dibrugarh": [27.4728, 94.9120],
    "Jorhat": [26.7509, 94.2037],
    "Tezpur": [26.6338, 92.7926],
    "Thrissur": [10.5276, 76.2144],
    "Kozhikode": [11.2588, 75.7804],
    "Kollam": [8.8932, 76.6141],
    "Palakkad": [10.7867, 76.6548],
    "Alappuzha": [9.4981, 76.3388],
    "Tirunelveli": [8.7139, 77.7567],
    "Thoothukudi": [8.7642, 78.1348],
    "Thanjavur": [10.7870, 79.1378],
    "Dindigul": [10.3673, 77.9803],
    "Nagercoil": [8.1833, 77.4119],
    "Pondicherry": [11.9416, 79.8083],
    "Margao": [15.2832, 73.9862],
    "Vasco da Gama": [15.3982, 73.8113],
}

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("=" * 70)
print("MAXIMUM ACCURACY PIPELINE")
print("=" * 70)

df = pd.read_csv("accident_prediction_india.csv")
print(f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns")

df["Traffic Control Presence"] = df["Traffic Control Presence"].fillna("None")
df["Driver License Status"] = df["Driver License Status"].fillna("Unknown")
print(f"Missing values after fill: {df.isnull().sum().sum()}")

# ============================================================
# STEP 2: AGGRESSIVE FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: FEATURE ENGINEERING (ALL POSSIBLE FEATURES)")
print("=" * 70)


def parse_hour(t):
    try:
        return int(str(t).split(":")[0])
    except Exception:
        return 12


df["hour"] = df["Time of Day"].apply(parse_hour)

# Time features
df["is_night"] = df["hour"].apply(lambda h: 1 if h >= 21 or h <= 5 else 0)
df["is_rush_hour"] = df["hour"].apply(lambda h: 1 if h in [8, 9, 17, 18, 19] else 0)
df["is_early_morning"] = df["hour"].apply(lambda h: 1 if 4 <= h <= 6 else 0)
df["is_late_night"] = df["hour"].apply(lambda h: 1 if 0 <= h <= 3 else 0)


def time_bin(h):
    if 0 <= h < 4:
        return "Late Night"
    elif 4 <= h < 7:
        return "Early Morning"
    elif 7 <= h < 10:
        return "Morning Rush"
    elif 10 <= h < 13:
        return "Late Morning"
    elif 13 <= h < 16:
        return "Afternoon"
    elif 16 <= h < 19:
        return "Evening Rush"
    elif 19 <= h < 21:
        return "Evening"
    else:
        return "Night"


df["time_bin"] = df["hour"].apply(time_bin)

# Weather features
df["rain_indicator"] = (df["Weather Conditions"] == "Rainy").astype(int)
df["fog_indicator"] = (df["Weather Conditions"] == "Foggy").astype(int)
df["storm_indicator"] = (df["Weather Conditions"] == "Stormy").astype(int)
df["bad_weather"] = (df["Weather Conditions"] != "Clear").astype(int)

# Day features
df["is_weekend"] = df["Day of Week"].isin(["Saturday", "Sunday"]).astype(int)
df["is_monday"] = (df["Day of Week"] == "Monday").astype(int)
df["is_friday"] = (df["Day of Week"] == "Friday").astype(int)

# Month/Season
month_map = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}
df["month_num"] = df["Month"].map(month_map)


def get_season(m):
    if m in [12, 1, 2]:
        return "Winter"
    elif m in [3, 4, 5]:
        return "Summer"
    elif m in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"


df["season"] = df["month_num"].apply(get_season)
df["is_monsoon"] = (df["season"] == "Monsoon").astype(int)
df["is_winter"] = (df["season"] == "Winter").astype(int)

# Road risk
road_type_risk = {
    "National Highway": 0.85,
    "State Highway": 0.70,
    "Urban Road": 0.45,
    "Village Road": 0.60,
}
road_cond_risk = {"Dry": 0.2, "Wet": 0.65, "Under Construction": 0.80, "Damaged": 0.90}
df["road_type_risk"] = df["Road Type"].map(road_type_risk).fillna(0.5)
df["road_cond_risk"] = df["Road Condition"].map(road_cond_risk).fillna(0.5)
df["road_risk_score"] = (df["road_type_risk"] * 0.5 + df["road_cond_risk"] * 0.5).round(
    4
)

# Speed features
df["speed_over_80"] = (df["Speed Limit (km/h)"] > 80).astype(int)
df["speed_over_100"] = (df["Speed Limit (km/h)"] > 100).astype(int)
df["speed_under_40"] = (df["Speed Limit (km/h)"] < 40).astype(int)
speed_max = df["Speed Limit (km/h)"].max()
df["speed_normalized"] = df["Speed Limit (km/h)"] / speed_max

# Traffic density
veh_max = df["Number of Vehicles Involved"].max()
df["vehicles_normalized"] = df["Number of Vehicles Involved"] / veh_max
df["traffic_density"] = (
    0.5 * df["vehicles_normalized"] + 0.5 * df["speed_normalized"]
).round(4)

# Driver features
df["young_driver"] = (df["Driver Age"] < 25).astype(int)
df["senior_driver"] = (df["Driver Age"] > 60).astype(int)
df["age_normalized"] = df["Driver Age"] / df["Driver Age"].max()
df["expired_license"] = (df["Driver License Status"] == "Expired").astype(int)
df["no_license_info"] = (df["Driver License Status"] == "Unknown").astype(int)
df["alcohol_yes"] = (df["Alcohol Involvement"] == "Yes").astype(int)

# Interaction features
df["night_rain"] = df["is_night"] * df["rain_indicator"]
df["night_speed_high"] = df["is_night"] * df["speed_over_80"]
df["alcohol_night"] = df["alcohol_yes"] * df["is_night"]
df["alcohol_speed_high"] = df["alcohol_yes"] * df["speed_over_80"]
df["young_alcohol"] = df["young_driver"] * df["alcohol_yes"]
df["bad_weather_night"] = df["bad_weather"] * df["is_night"]
df["highway_speed"] = (df["Road Type"] == "National Highway").astype(int) * df[
    "speed_normalized"
]
df["damaged_road_speed"] = (df["Road Condition"] == "Damaged").astype(int) * df[
    "speed_normalized"
]

# Casualty features (strong severity predictors)
df["has_fatalities"] = (df["Number of Fatalities"] > 0).astype(int)
df["high_casualties"] = (df["Number of Casualties"] >= 5).astype(int)
df["casualty_fatality_ratio"] = np.where(
    df["Number of Casualties"] > 0,
    df["Number of Fatalities"] / df["Number of Casualties"],
    0,
).round(4)

print(f"Total features engineered: {len(df.columns) - 22} new features")

# ============================================================
# STEP 3: DEFINE FEATURES AND TARGET
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: DEFINING FEATURES & TARGET")
print("=" * 70)

cat_features = [
    "Vehicle Type Involved",
    "Weather Conditions",
    "Road Type",
    "Lighting Conditions",
    "Road Condition",
    "Alcohol Involvement",
    "Driver Gender",
    "Traffic Control Presence",
    "Driver License Status",
    "Accident Location Details",
    "time_bin",
    "season",
    "State Name",
    "Day of Week",
    "Month",
]

num_features = [
    "Speed Limit (km/h)",
    "Driver Age",
    "Number of Vehicles Involved",
    "Number of Casualties",
    "Number of Fatalities",
    "hour",
    "month_num",
    "rain_indicator",
    "fog_indicator",
    "storm_indicator",
    "bad_weather",
    "is_night",
    "is_rush_hour",
    "is_early_morning",
    "is_late_night",
    "is_weekend",
    "is_monday",
    "is_friday",
    "is_monsoon",
    "is_winter",
    "road_type_risk",
    "road_cond_risk",
    "road_risk_score",
    "speed_over_80",
    "speed_over_100",
    "speed_under_40",
    "speed_normalized",
    "vehicles_normalized",
    "traffic_density",
    "young_driver",
    "senior_driver",
    "age_normalized",
    "expired_license",
    "no_license_info",
    "alcohol_yes",
    "night_rain",
    "night_speed_high",
    "alcohol_night",
    "alcohol_speed_high",
    "young_alcohol",
    "bad_weather_night",
    "highway_speed",
    "damaged_road_speed",
    "has_fatalities",
    "high_casualties",
    "casualty_fatality_ratio",
]

feature_cols = cat_features + num_features
X = df[feature_cols]
y = df["Accident Severity"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"Categorical features: {len(cat_features)}")
print(f"Numeric features: {len(num_features)}")
print(f"Total input features: {len(feature_cols)}")
print(f"Classes: {list(le.classes_)}")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_features,
        ),
        ("num", StandardScaler(), num_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)

print(f"Training: {X_train.shape[0]} | Test: {X_test.shape[0]}")

X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()
print(f"Total encoded features: {len(feature_names)}")

# ============================================================
# STEP 4: TRAIN ALL MODELS
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: TRAINING MODELS (TUNED FOR MAX ACCURACY)")
print("=" * 70)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, C=1.0, solver="lbfgs", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_split=5,
        random_state=42,
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

    print(f"  Accuracy:  {acc:.4f} ({acc:.2%})")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print(f"  Confusion Matrix:")
    print("  " + cm_df.to_string().replace("\n", "\n  "))

    print(f"  Classification Report:")
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    print("  " + report.replace("\n", "\n  "))

# ============================================================
# STEP 5: STACKING ENSEMBLE (MAX ACCURACY)
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: STACKING ENSEMBLE")
print("=" * 70)

estimators = [
    ("rf", trained_models["Random Forest"]),
    ("et", trained_models["Extra Trees"]),
    ("xgb", trained_models["XGBoost"]),
    ("gb", trained_models["Gradient Boosting"]),
]

stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=2000, random_state=42),
    cv=5,
    n_jobs=-1,
    passthrough=False,
)

print("Training Stacking Ensemble (RF + ET + XGB + GB -> LR)...")
stacking_model.fit(X_train_enc, y_train)
y_pred_stack = stacking_model.predict(X_test_enc)

acc_stack = accuracy_score(y_test, y_pred_stack)
prec_stack = precision_score(y_test, y_pred_stack, average="weighted")
rec_stack = recall_score(y_test, y_pred_stack, average="weighted")
f1_stack = f1_score(y_test, y_pred_stack, average="weighted")

results["Stacking Ensemble"] = {
    "Accuracy": acc_stack,
    "Precision": prec_stack,
    "Recall": rec_stack,
    "F1 Score": f1_stack,
}
trained_models["Stacking Ensemble"] = stacking_model

print(f"  Accuracy:  {acc_stack:.4f} ({acc_stack:.2%})")
print(f"  Precision: {prec_stack:.4f}")
print(f"  Recall:    {rec_stack:.4f}")
print(f"  F1 Score:  {f1_stack:.4f}")

cm_stack = confusion_matrix(y_test, y_pred_stack)
cm_df = pd.DataFrame(cm_stack, index=le.classes_, columns=le.classes_)
print(f"  Confusion Matrix:")
print("  " + cm_df.to_string().replace("\n", "\n  "))
print(f"  Classification Report:")
report = classification_report(y_test, y_pred_stack, target_names=le.classes_)
print("  " + report.replace("\n", "\n  "))

# Voting Ensemble
print("\n--- Soft Voting Ensemble ---")
voting_model = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
voting_model.fit(X_train_enc, y_train)
y_pred_vote = voting_model.predict(X_test_enc)

acc_vote = accuracy_score(y_test, y_pred_vote)
f1_vote = f1_score(y_test, y_pred_vote, average="weighted")
results["Voting Ensemble"] = {
    "Accuracy": acc_vote,
    "Precision": precision_score(y_test, y_pred_vote, average="weighted"),
    "Recall": recall_score(y_test, y_pred_vote, average="weighted"),
    "F1 Score": f1_vote,
}
trained_models["Voting Ensemble"] = voting_model
print(f"  Accuracy:  {acc_vote:.4f} ({acc_vote:.2%})")
print(f"  F1 Score:  {f1_vote:.4f}")

# ============================================================
# MODEL COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("MODEL COMPARISON TABLE (SORTED BY ACCURACY)")
print("=" * 70)

comparison_df = pd.DataFrame(results).T.sort_values("Accuracy", ascending=False)
print(f"\n{comparison_df.to_string()}")

best_model_name = comparison_df.index[0]
best_model = trained_models[best_model_name]
best_acc = comparison_df.iloc[0]["Accuracy"]
print(f"\n>>> BEST MODEL: {best_model_name} (Accuracy={best_acc:.4f}, {best_acc:.2%})")

# Comparison chart
fig, ax = plt.subplots(figsize=(14, 7))
comparison_df.plot(kind="bar", ax=ax, colormap="viridis")
ax.set_title(
    "Model Comparison: All Metrics (Max Accuracy Pipeline)",
    fontsize=14,
    fontweight="bold",
)
ax.set_ylabel("Score")
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
ax.legend(loc="lower right")
ax.set_ylim(0, 1)
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", fontsize=7, padding=2)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: model_comparison.png")

# ============================================================
# STEP 6: FEATURE IMPORTANCE & SHAP
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: FEATURE IMPORTANCE & SHAP")
print("=" * 70)

rf_model = trained_models["Random Forest"]
rf_imp = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(
    ascending=False
)

print("\n--- Top 20 Features (Random Forest) ---")
for i, (feat, imp) in enumerate(rf_imp.head(20).items(), 1):
    print(f"  {i:2d}. {feat}: {imp:.4f} ({imp:.1%})")

fig, ax = plt.subplots(figsize=(14, 10))
top20 = rf_imp.head(20).sort_values(ascending=True)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top20)))
ax.barh(top20.index, top20.values, color=colors)
ax.set_title(
    "Top 20 Feature Importances (Random Forest)", fontsize=14, fontweight="bold"
)
ax.set_xlabel("Importance")
for i, v in enumerate(top20.values):
    ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: feature_importance.png")

# SHAP
print("\n--- Computing SHAP Values (XGBoost) ---")
xgb_model = trained_models["XGBoost"]
explainer = shap.TreeExplainer(xgb_model)
shap_sample = X_test_enc[:200]
shap_values = explainer.shap_values(shap_sample)

if isinstance(shap_values, list):
    mean_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
else:
    mean_shap = (
        np.abs(shap_values).mean(axis=(0, 2))
        if shap_values.ndim == 3
        else np.abs(shap_values).mean(axis=0)
    )

shap_imp = pd.Series(mean_shap.mean(axis=0), index=feature_names).sort_values(
    ascending=False
)

print("\n--- Top Factors Influencing Accidents (SHAP) ---")
total_shap = shap_imp.sum()
for feat, val in shap_imp.head(10).items():
    print(f"  {feat}: {val / total_shap * 100:.1f}%")

fig, ax = plt.subplots(figsize=(14, 10))
top20_shap = shap_imp.head(20).sort_values(ascending=True)
colors_s = plt.cm.magma(np.linspace(0.3, 0.85, len(top20_shap)))
ax.barh(top20_shap.index, top20_shap.values, color=colors_s)
ax.set_title("Top 20 Features by SHAP Importance", fontsize=14, fontweight="bold")
ax.set_xlabel("Mean |SHAP Value|")
plt.tight_layout()
plt.savefig("shap_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_importance.png")

try:
    fig = plt.figure(figsize=(14, 8))
    shap.summary_plot(
        shap_values,
        shap_sample,
        feature_names=list(feature_names),
        plot_type="bar",
        show=False,
        max_display=20,
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
    time_of_day,
    month,
    state,
    city,
    location_detail,
    driver_license,
    num_casualties=0,
    num_fatalities=0,
    model=best_model,
    preproc=preprocessor,
    label_enc=le,
):
    hour = parse_hour(time_of_day)
    tbin = time_bin(hour)
    m_num = month_map.get(month, 6)
    ssn = get_season(m_num)

    rain = 1 if weather == "Rainy" else 0
    fog = 1 if weather == "Foggy" else 0
    storm = 1 if weather == "Stormy" else 0
    bad_w = 0 if weather == "Clear" else 1
    night = 1 if hour >= 21 or hour <= 5 else 0
    rush = 1 if hour in [8, 9, 17, 18, 19] else 0
    early = 1 if 4 <= hour <= 6 else 0
    late_n = 1 if 0 <= hour <= 3 else 0
    wknd = 1 if day_of_week in ["Saturday", "Sunday"] else 0
    mon = 1 if day_of_week == "Monday" else 0
    fri = 1 if day_of_week == "Friday" else 0
    monsoon = 1 if ssn == "Monsoon" else 0
    winter = 1 if ssn == "Winter" else 0
    rt_risk = road_type_risk.get(road_type, 0.5)
    rc_risk = road_cond_risk.get(road_condition, 0.5)
    rr_score = round(rt_risk * 0.5 + rc_risk * 0.5, 4)
    s80 = 1 if speed > 80 else 0
    s100 = 1 if speed > 100 else 0
    s40 = 1 if speed < 40 else 0
    s_norm = speed / speed_max
    v_norm = num_vehicles / veh_max
    t_dens = round(0.5 * v_norm + 0.5 * s_norm, 4)
    young = 1 if driver_age < 25 else 0
    senior = 1 if driver_age > 60 else 0
    age_n = driver_age / 70
    exp_lic = 1 if driver_license == "Expired" else 0
    no_lic = 1 if driver_license == "Unknown" else 0
    alc = 1 if alcohol == "Yes" else 0
    has_fat = 1 if num_fatalities > 0 else 0
    hi_cas = 1 if num_casualties >= 5 else 0
    cf_ratio = round(num_fatalities / num_casualties, 4) if num_casualties > 0 else 0

    input_data = pd.DataFrame(
        [
            {
                "Vehicle Type Involved": vehicle_type,
                "Weather Conditions": weather,
                "Road Type": road_type,
                "Lighting Conditions": lighting,
                "Road Condition": road_condition,
                "Alcohol Involvement": alcohol,
                "Driver Gender": driver_gender,
                "Traffic Control Presence": traffic_control,
                "Driver License Status": driver_license,
                "Accident Location Details": location_detail,
                "time_bin": tbin,
                "season": ssn,
                "State Name": state,
                "Day of Week": day_of_week,
                "Month": month,
                "Speed Limit (km/h)": speed,
                "Driver Age": driver_age,
                "Number of Vehicles Involved": num_vehicles,
                "Number of Casualties": num_casualties,
                "Number of Fatalities": num_fatalities,
                "hour": hour,
                "month_num": m_num,
                "rain_indicator": rain,
                "fog_indicator": fog,
                "storm_indicator": storm,
                "bad_weather": bad_w,
                "is_night": night,
                "is_rush_hour": rush,
                "is_early_morning": early,
                "is_late_night": late_n,
                "is_weekend": wknd,
                "is_monday": mon,
                "is_friday": fri,
                "is_monsoon": monsoon,
                "is_winter": winter,
                "road_type_risk": rt_risk,
                "road_cond_risk": rc_risk,
                "road_risk_score": rr_score,
                "speed_over_80": s80,
                "speed_over_100": s100,
                "speed_under_40": s40,
                "speed_normalized": s_norm,
                "vehicles_normalized": v_norm,
                "traffic_density": t_dens,
                "young_driver": young,
                "senior_driver": senior,
                "age_normalized": age_n,
                "expired_license": exp_lic,
                "no_license_info": no_lic,
                "alcohol_yes": alc,
                "night_rain": night * rain,
                "night_speed_high": night * s80,
                "alcohol_night": alc * night,
                "alcohol_speed_high": alc * s80,
                "young_alcohol": young * alc,
                "bad_weather_night": bad_w * night,
                "highway_speed": (1 if road_type == "National Highway" else 0) * s_norm,
                "damaged_road_speed": (1 if road_condition == "Damaged" else 0)
                * s_norm,
                "has_fatalities": has_fat,
                "high_casualties": hi_cas,
                "casualty_fatality_ratio": cf_ratio,
            }
        ]
    )

    X_enc = preproc.transform(input_data)
    probs = model.predict_proba(X_enc)[0]
    pred_idx = np.argmax(probs)
    pred_class = label_enc.classes_[pred_idx]
    fatal_idx = list(label_enc.classes_).index("Fatal")
    fatal_prob = probs[fatal_idx]

    if fatal_prob >= 0.5:
        risk_level = "HIGH"
    elif fatal_prob >= 0.3:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    return {
        "predicted_severity": pred_class,
        "confidence": round(float(probs[pred_idx]), 4),
        "probabilities": {
            label_enc.classes_[i]: round(float(p), 4) for i, p in enumerate(probs)
        },
        "accident_risk": round(float(fatal_prob), 4),
        "risk_level": risk_level,
    }


print("\n--- Example Prediction: High Risk ---")
r1 = predict_accident_risk(
    weather="Rainy",
    road_type="National Highway",
    road_condition="Wet",
    lighting="Dark",
    vehicle_type="Two-Wheeler",
    speed=95,
    alcohol="Yes",
    driver_age=21,
    driver_gender="Male",
    traffic_control="None",
    num_vehicles=3,
    day_of_week="Saturday",
    time_of_day="23:30",
    month="July",
    state="Uttar Pradesh",
    city="Lucknow",
    location_detail="Curve",
    driver_license="Expired",
    num_casualties=4,
    num_fatalities=2,
)
print(f"  Severity: {r1['predicted_severity']} | Confidence: {r1['confidence']:.2%}")
print(f"  Risk: {r1['accident_risk']:.2f} | Level: {r1['risk_level']}")

print("\n--- Example Prediction: Low Risk ---")
r2 = predict_accident_risk(
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
    time_of_day="10:00",
    month="March",
    state="Karnataka",
    city="Bangalore",
    location_detail="Straight Road",
    driver_license="Valid",
    num_casualties=1,
    num_fatalities=0,
)
print(f"  Severity: {r2['predicted_severity']} | Confidence: {r2['confidence']:.2%}")
print(f"  Risk: {r2['accident_risk']:.2f} | Level: {r2['risk_level']}")

# ============================================================
# STEP 8: COMPREHENSIVE RISK MAP (ALL INDIA)
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: ACCIDENT RISK MAP (ALL INDIA LOCATIONS)")
print("=" * 70)

severity_map = {"Minor": 1, "Serious": 2, "Fatal": 3}
df["Severity_Score"] = df["Accident Severity"].map(severity_map)

# State-level risk
state_risk = (
    df.groupby("State Name")
    .agg(
        avg_severity=("Severity_Score", "mean"),
        total_accidents=("Severity_Score", "count"),
        fatal_count=("Accident Severity", lambda x: (x == "Fatal").sum()),
        serious_count=("Accident Severity", lambda x: (x == "Serious").sum()),
        avg_speed=("Speed Limit (km/h)", "mean"),
        alcohol_rate=("alcohol_yes", "mean"),
    )
    .reset_index()
)
state_risk["fatal_rate"] = state_risk["fatal_count"] / state_risk["total_accidents"]

# City-level risk
city_risk = (
    df[df["City Name"] != "Unknown"]
    .groupby(["City Name", "State Name"])
    .agg(
        avg_severity=("Severity_Score", "mean"),
        total_accidents=("Severity_Score", "count"),
        fatal_count=("Accident Severity", lambda x: (x == "Fatal").sum()),
        avg_speed=("Speed Limit (km/h)", "mean"),
    )
    .reset_index()
)
city_risk["fatal_rate"] = city_risk["fatal_count"] / city_risk["total_accidents"]

# Create map
india_map = folium.Map(location=[22.0, 79.0], zoom_start=5, tiles="cartodbpositron")

# Heatmap
heat_data = []
for _, row in state_risk.iterrows():
    coords = STATE_COORDS.get(row["State Name"])
    if coords:
        weight = (
            row["avg_severity"]
            * row["total_accidents"]
            / state_risk["total_accidents"].max()
        )
        heat_data.append([coords[0], coords[1], weight])

for _, row in city_risk.iterrows():
    coords = CITY_COORDS.get(row["City Name"])
    if coords:
        weight = (
            row["avg_severity"]
            * row["total_accidents"]
            / max(city_risk["total_accidents"].max(), 1)
        )
        heat_data.append([coords[0], coords[1], weight * 0.5])

HeatMap(
    heat_data,
    radius=35,
    blur=25,
    max_zoom=7,
    gradient={
        "0.2": "blue",
        "0.4": "lime",
        "0.6": "yellow",
        "0.8": "orange",
        "1.0": "red",
    },
).add_to(india_map)

# State markers
state_group = folium.FeatureGroup(name="State Risk Zones")
for _, row in state_risk.iterrows():
    coords = STATE_COORDS.get(row["State Name"])
    if not coords:
        continue
    sev = row["avg_severity"]
    if sev >= 2.15:
        color, level = "red", "HIGH RISK"
    elif sev >= 1.9:
        color, level = "orange", "MODERATE"
    else:
        color, level = "green", "LOW RISK"

    popup_html = (
        f"<div style='font-family:Arial;width:220px;'>"
        f"<h4 style='margin:0;color:{color};'>{row['State Name']}</h4>"
        f"<hr style='margin:3px 0;'>"
        f"<b>Risk Level:</b> <span style='color:{color};font-weight:bold;'>{level}</span><br>"
        f"<b>Total Accidents:</b> {row['total_accidents']}<br>"
        f"<b>Fatal:</b> {row['fatal_count']} ({row['fatal_rate']:.0%})<br>"
        f"<b>Serious:</b> {row['serious_count']}<br>"
        f"<b>Avg Severity:</b> {sev:.2f}/3.00<br>"
        f"<b>Avg Speed:</b> {row['avg_speed']:.0f} km/h<br>"
        f"<b>Alcohol Rate:</b> {row['alcohol_rate']:.0%}"
        f"</div>"
    )
    folium.CircleMarker(
        location=coords,
        radius=max(6, row["total_accidents"] / 6),
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=folium.Popup(popup_html, max_width=250),
        tooltip=f"{row['State Name']}: {level} ({row['total_accidents']} accidents)",
    ).add_to(state_group)
state_group.add_to(india_map)

# City markers
city_group = folium.FeatureGroup(name="City Details")
for _, row in city_risk.iterrows():
    coords = CITY_COORDS.get(row["City Name"])
    if not coords:
        continue
    sev = row["avg_severity"]
    if sev >= 2.15:
        color = "red"
    elif sev >= 1.9:
        color = "orange"
    else:
        color = "green"

    popup_html = (
        f"<div style='font-family:Arial;width:200px;'>"
        f"<h4 style='margin:0;'>{row['City Name']}</h4>"
        f"<small>{row['State Name']}</small>"
        f"<hr style='margin:3px 0;'>"
        f"<b>Accidents:</b> {row['total_accidents']}<br>"
        f"<b>Fatal:</b> {row['fatal_count']} ({row['fatal_rate']:.0%})<br>"
        f"<b>Severity:</b> {sev:.2f}/3<br>"
        f"<b>Avg Speed:</b> {row['avg_speed']:.0f} km/h"
        f"</div>"
    )
    folium.Marker(
        location=coords,
        popup=folium.Popup(popup_html, max_width=220),
        tooltip=f"{row['City Name']}: {row['total_accidents']} accidents",
        icon=folium.Icon(color=color, icon="info-sign"),
    ).add_to(city_group)
city_group.add_to(india_map)

folium.LayerControl().add_to(india_map)

legend_html = """
<div style="position:fixed;bottom:50px;left:50px;z-index:1000;
background:white;padding:12px 16px;border-radius:8px;border:2px solid #555;
font-size:13px;font-family:Arial;box-shadow:2px 2px 6px rgba(0,0,0,0.3);">
<b>Accident Risk Level</b><br><br>
<span style="background:red;width:14px;height:14px;display:inline-block;border-radius:50%;vertical-align:middle;"></span>
 High Risk (Severity >= 2.15)<br>
<span style="background:orange;width:14px;height:14px;display:inline-block;border-radius:50%;vertical-align:middle;"></span>
 Moderate (1.9 - 2.15)<br>
<span style="background:green;width:14px;height:14px;display:inline-block;border-radius:50%;vertical-align:middle;"></span>
 Low Risk (< 1.9)<br><br>
<small>Circle size = accident count<br>
Heatmap intensity = severity x volume</small>
</div>
"""
india_map.get_root().html.add_child(folium.Element(legend_html))

india_map.save("accident_risk_map.html")
print("Saved: accident_risk_map.html")
print(
    f"States mapped: {sum(1 for s in state_risk['State Name'] if s in STATE_COORDS)}/{len(state_risk)}"
)
print(
    f"Cities mapped: {sum(1 for c in city_risk['City Name'] if c in CITY_COORDS)}/{len(city_risk)}"
)

print("\n--- State Risk Zones ---")
for _, row in state_risk.sort_values("avg_severity", ascending=False).iterrows():
    sev = row["avg_severity"]
    zone = "RED" if sev >= 2.15 else ("YELLOW" if sev >= 1.9 else "GREEN")
    print(
        f"  {row['State Name']:25s} | Sev: {sev:.2f} | Fatal: {row['fatal_rate']:.0%} | Accidents: {row['total_accidents']:3d} | {zone}"
    )

print("\n--- City Risk Zones ---")
for _, row in city_risk.sort_values("avg_severity", ascending=False).iterrows():
    sev = row["avg_severity"]
    zone = "RED" if sev >= 2.15 else ("YELLOW" if sev >= 1.9 else "GREEN")
    print(
        f"  {row['City Name']:20s} ({row['State Name']:18s}) | Sev: {sev:.2f} | Fatal: {row['fatal_rate']:.0%} | {zone}"
    )

# ============================================================
# STEP 9: SAVE MODEL
# ============================================================
print("\n" + "=" * 70)
print("STEP 9: SAVING MODELS & ARTIFACTS")
print("=" * 70)

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
    "month_map": month_map,
    "speed_max": speed_max,
    "veh_max": veh_max,
    "results": results,
    "state_coords": STATE_COORDS,
    "city_coords": CITY_COORDS,
}

joblib.dump(artifacts, "model.pkl")
joblib.dump(best_model, "accident_severity_model.pkl")
print("Saved: model.pkl (full pipeline)")
print("Saved: accident_severity_model.pkl (best model)")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("PIPELINE COMPLETE -- FINAL SUMMARY")
print("=" * 70)

print(f"\n  Best Model: {best_model_name}")
print(
    f"  Accuracy:   {results[best_model_name]['Accuracy']:.4f} ({results[best_model_name]['Accuracy']:.2%})"
)
print(f"  F1 Score:   {results[best_model_name]['F1 Score']:.4f}")

print(f"\n  All Model Results:")
for name, metrics in sorted(
    results.items(), key=lambda x: x[1]["Accuracy"], reverse=True
):
    print(
        f"    {name:25s} -> Accuracy: {metrics['Accuracy']:.4f} | F1: {metrics['F1 Score']:.4f}"
    )

print(
    f"\n  Features: {len(feature_cols)} input ({len(cat_features)} cat + {len(num_features)} num)"
)
print(f"  Encoded:  {len(feature_names)} total after OneHotEncoding")

print(f"\n  Location Database:")
print(f"    States/UTs: {len(STATE_COORDS)} coordinates")
print(f"    Cities:     {len(CITY_COORDS)} coordinates")

print(f"\n  Output Files:")
print(f"    model.pkl                   - Full pipeline artifact")
print(f"    accident_severity_model.pkl - Best model")
print(f"    model_comparison.png        - Metric comparison chart")
print(f"    feature_importance.png      - RF feature importance")
print(f"    shap_importance.png         - SHAP importance")
print(f"    shap_bar_by_class.png       - SHAP by severity class")
print(f"    accident_risk_map.html      - Interactive India risk map")
print(f"\n{'=' * 70}")
