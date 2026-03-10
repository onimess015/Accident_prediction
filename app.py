"""
Accident Risk Prediction Dashboard
Step 8: Interactive Streamlit dashboard with weather integration (Step 10)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Accident Risk Predictor",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Accident Severity Prediction Dashboard")
st.markdown("Predict accident severity based on road, weather, and driver conditions.")


# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_artifacts():
    return joblib.load("model.pkl")


try:
    artifacts = load_artifacts()
    model = artifacts["best_model"]
    preprocessor = artifacts["preprocessor"]
    le = artifacts["label_encoder"]
    road_type_risk = artifacts["road_type_risk"]
    road_cond_risk = artifacts["road_cond_risk"]
    model_results = artifacts["results"]
    best_model_name = artifacts["best_model_name"]
except FileNotFoundError:
    st.error("model.pkl not found. Run enhanced_pipeline.py first.")
    st.stop()

# ============================================================
# SIDEBAR — USER INPUTS
# ============================================================
st.sidebar.header("Input Accident Scenario")

weather = st.sidebar.selectbox(
    "Weather Condition", ["Clear", "Rainy", "Foggy", "Hazy", "Windy"]
)
time_lighting = st.sidebar.selectbox(
    "Lighting / Time of Day", ["Daylight", "Dark", "Dawn", "Dusk"]
)
road_type = st.sidebar.selectbox(
    "Road Type",
    ["National Highway", "State Highway", "Urban Road", "Rural Road", "Expressway"],
)
road_condition = st.sidebar.selectbox(
    "Road Condition", ["Dry", "Wet", "Under Construction", "Flooded", "Icy"]
)
vehicle_type = st.sidebar.selectbox(
    "Vehicle Type",
    ["Car", "Two-Wheeler", "Truck", "Bus", "Auto-Rickshaw", "Cycle", "Pedestrian"],
)
speed = st.sidebar.slider("Speed Limit (km/h)", 20, 120, 60)
alcohol = st.sidebar.selectbox("Alcohol Involvement", ["No", "Yes"])
driver_age = st.sidebar.slider("Driver Age", 18, 80, 35)
driver_gender = st.sidebar.selectbox("Driver Gender", ["Male", "Female"])
traffic_control = st.sidebar.selectbox(
    "Traffic Control", ["Signals", "Signs", "Police Checkpost", "None", "Unknown"]
)
num_vehicles = st.sidebar.slider("Number of Vehicles", 1, 10, 2)
day_of_week = st.sidebar.selectbox(
    "Day of Week",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
)

# ============================================================
# STEP 10: REAL-TIME WEATHER INTEGRATION
# ============================================================
st.sidebar.markdown("---")
st.sidebar.header("Live Weather (Optional)")
city_input = st.sidebar.text_input("Enter city name for live weather", "")

live_weather = None
if city_input:
    try:
        # Using wttr.in free API (no key needed)
        resp = requests.get(
            f"https://wttr.in/{city_input}?format=j1",
            timeout=5,
            headers={"User-Agent": "AccidentPredictor/1.0"},
        )
        if resp.status_code == 200:
            data = resp.json()
            current = data["current_condition"][0]
            temp_c = current["temp_C"]
            humidity = current["humidity"]
            desc = current["weatherDesc"][0]["value"]
            wind_speed = current["windspeedKmph"]

            live_weather = {
                "temp": temp_c,
                "humidity": humidity,
                "description": desc,
                "wind_speed": wind_speed,
            }

            st.sidebar.success(f"Weather for {city_input}")
            st.sidebar.metric("Temperature", f"{temp_c} C")
            st.sidebar.metric("Humidity", f"{humidity}%")
            st.sidebar.metric("Condition", desc)
            st.sidebar.metric("Wind", f"{wind_speed} km/h")

            # Auto-map weather to model input
            desc_lower = desc.lower()
            if (
                "rain" in desc_lower
                or "drizzle" in desc_lower
                or "shower" in desc_lower
            ):
                weather = "Rainy"
            elif "fog" in desc_lower or "mist" in desc_lower:
                weather = "Foggy"
            elif "haze" in desc_lower or "smoke" in desc_lower:
                weather = "Hazy"
            elif "wind" in desc_lower or "gale" in desc_lower:
                weather = "Windy"
            else:
                weather = "Clear"

            st.sidebar.info(f"Auto-mapped to: **{weather}**")
        else:
            st.sidebar.warning("Could not fetch weather data.")
    except Exception:
        st.sidebar.warning("Weather service unavailable.")


# ============================================================
# PREDICTION
# ============================================================
def predict():
    rain = 1 if weather == "Rainy" else 0
    night = 1 if time_lighting in ["Dark", "Dawn", "Dusk"] else 0
    weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0
    speed_n = speed / 120
    vehicles_n = num_vehicles / 10
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
                "Lighting Conditions": time_lighting,
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
    predicted_class = le.classes_[predicted_class_idx]

    fatal_idx = list(le.classes_).index("Fatal")
    fatal_prob = probabilities[fatal_idx]

    if fatal_prob >= 0.5:
        risk_level = "HIGH"
        risk_color = "red"
    elif fatal_prob >= 0.3:
        risk_level = "MODERATE"
        risk_color = "orange"
    else:
        risk_level = "LOW"
        risk_color = "green"

    return {
        "predicted": predicted_class,
        "confidence": probabilities[predicted_class_idx],
        "probs": {le.classes_[i]: float(p) for i, p in enumerate(probabilities)},
        "fatal_prob": fatal_prob,
        "risk_level": risk_level,
        "risk_color": risk_color,
    }


result = predict()

# ============================================================
# DISPLAY RESULTS
# ============================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Predicted Severity", result["predicted"])
with col2:
    st.metric("Confidence", f"{result['confidence']:.1%}")
with col3:
    color_map = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}
    st.metric(
        "Risk Level",
        f"{color_map.get(result['risk_level'], '')} {result['risk_level']}",
    )

st.markdown("---")

# Probability gauge
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Severity Probabilities")
    probs = result["probs"]

    fig_bar = go.Figure()
    colors = {"Fatal": "#e74c3c", "Serious": "#f39c12", "Minor": "#2ecc71"}
    for cls in ["Fatal", "Serious", "Minor"]:
        fig_bar.add_trace(
            go.Bar(
                x=[cls],
                y=[probs[cls]],
                marker_color=colors[cls],
                name=cls,
                text=f"{probs[cls]:.1%}",
                textposition="auto",
            )
        )
    fig_bar.update_layout(
        yaxis_title="Probability", yaxis_range=[0, 1], showlegend=False, height=350
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_b:
    st.subheader("Accident Risk Gauge")
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=result["fatal_prob"] * 100,
            title={"text": "Fatal Accident Risk (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": result["risk_color"]},
                "steps": [
                    {"range": [0, 30], "color": "#d4edda"},
                    {"range": [30, 50], "color": "#fff3cd"},
                    {"range": [50, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 2},
                    "thickness": 0.75,
                    "value": result["fatal_prob"] * 100,
                },
            },
        )
    )
    fig_gauge.update_layout(height=350)
    st.plotly_chart(fig_gauge, use_container_width=True)

# ============================================================
# FEATURE IMPORTANCE EXPLANATION
# ============================================================
st.markdown("---")
st.subheader("Feature Importance Explanation")

col_fi1, col_fi2 = st.columns(2)

with col_fi1:
    st.markdown("**Random Forest Feature Importance**")
    try:
        st.image("feature_importance.png")
    except Exception:
        st.info("Run enhanced_pipeline.py to generate feature_importance.png")

with col_fi2:
    st.markdown("**SHAP Feature Importance**")
    try:
        st.image("shap_importance.png")
    except Exception:
        st.info("Run enhanced_pipeline.py to generate shap_importance.png")

# ============================================================
# MODEL COMPARISON
# ============================================================
st.markdown("---")
st.subheader("Model Performance Comparison")

comparison_df = pd.DataFrame(model_results).T.sort_values("F1 Score", ascending=False)
comparison_df.index.name = "Model"

col_m1, col_m2 = st.columns([1, 2])

with col_m1:
    st.dataframe(
        comparison_df.style.format("{:.4f}").highlight_max(axis=0, color="#d4edda")
    )
    st.caption(f"Best model: **{best_model_name}**")

with col_m2:
    fig_comp = px.bar(
        comparison_df.reset_index().melt(id_vars="Model"),
        x="Model",
        y="value",
        color="variable",
        barmode="group",
        labels={"value": "Score", "variable": "Metric"},
        title="Model Comparison",
    )
    fig_comp.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig_comp, use_container_width=True)

# ============================================================
# RISK MAP
# ============================================================
st.markdown("---")
st.subheader("Accident Risk Map - India")

try:
    with open("accident_risk_map.html", "r", encoding="utf-8") as f:
        map_html = f.read()
    st.components.v1.html(map_html, height=600, scrolling=True)
except FileNotFoundError:
    st.info("Run enhanced_pipeline.py to generate accident_risk_map.html")

# ============================================================
# INPUT SUMMARY
# ============================================================
st.markdown("---")
st.subheader("Current Input Summary")
input_summary = {
    "Weather": weather,
    "Lighting": time_lighting,
    "Road Type": road_type,
    "Road Condition": road_condition,
    "Vehicle": vehicle_type,
    "Speed (km/h)": speed,
    "Alcohol": alcohol,
    "Driver Age": driver_age,
    "Gender": driver_gender,
    "Traffic Control": traffic_control,
    "Vehicles": num_vehicles,
    "Day": day_of_week,
}
if live_weather:
    input_summary["Live Weather"] = live_weather["description"]
    input_summary["Temperature"] = f"{live_weather['temp']} C"

st.json(input_summary)

st.markdown("---")
st.caption(
    "Built with Streamlit | Model trained on accident_prediction_india.csv (3000 records)"
)
