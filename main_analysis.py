import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")

print("=" * 70)
print("  INDIA ACCIDENT PREDICTION - COMPLETE ANALYSIS")
print("=" * 70)

# ================================================================
# STEP 1: LOAD ALL DATASETS
# ================================================================
print("\n[1/6] Loading all datasets...")

main_df = pd.read_csv("accident_prediction_india.csv")
cases_df = pd.read_csv("cases-reported-persons-injured-and-died.csv")
railway_cause_df = pd.read_csv("cause-wise-distribution-of-railway-accidents.csv")
road_cause_df = pd.read_csv(
    "cause-wise-distribution-of-road-accidents-and-unmanned-railway-crossing-accidents.csv"
)
railway_class_df = pd.read_csv("classification-of-railway-accidents.csv")
transport_2018_df = pd.read_csv(
    "mode-of-transport-wise-number-of-persons-died-in-road-accidents-2018-2020.csv"
)
transport_2021_df = pd.read_csv(
    "mode-of-transport-wise-number-of-persons-died-in-road-accidents-onwards-2021.csv"
)
month_df = pd.read_csv("month-of-occurrence-wise-number-of-traffic-accidents.csv")
place_df = pd.read_csv("place-of-occurrence-wise-road-accident-deaths.csv")
road_class_df = pd.read_csv(
    "road-classification-wise-number-of-road-accidents-injuries-and-deaths.csv"
)
time_df = pd.read_csv("time-of-occurrence-wise-number-of-traffic-accidents.csv")

print(f"  Main dataset: {main_df.shape[0]} rows x {main_df.shape[1]} columns")
print(f"  Total CSV files loaded: 11")

# ================================================================
# STEP 2: DATA CLEANING & PREPROCESSING
# ================================================================
print("\n[2/6] Cleaning data...")

# Binary indicators for missing data
main_df["Traffic_Control_Missing"] = (
    main_df["Traffic Control Presence"].isnull().astype(int)
)
main_df["License_Status_Missing"] = (
    main_df["Driver License Status"].isnull().astype(int)
)

# Fill missing values
main_df["Traffic Control Presence"] = main_df["Traffic Control Presence"].fillna(
    "Unknown"
)
main_df["Driver License Status"] = main_df["Driver License Status"].fillna("Unknown")

# Severity score for analysis
severity_map = {"Minor": 1, "Serious": 2, "Fatal": 3}
main_df["Severity_Score"] = main_df["Accident Severity"].map(severity_map)

null_count = main_df.isnull().sum().sum()
print(f"  Missing values after cleaning: {null_count}")
print(f"  Columns after preprocessing: {main_df.shape[1]}")

# ================================================================
# STEP 3: BASIC VISUALIZATIONS
# ================================================================
print("\n[3/6] Generating basic visualizations...")

# --- 3a. Accidents by Weather ---
fig, ax = plt.subplots(figsize=(10, 6))
weather_counts = main_df["Weather Conditions"].value_counts()
ax.barh(
    weather_counts.index,
    weather_counts.values,
    color=sns.color_palette("coolwarm", len(weather_counts)),
)
ax.set_title("Accidents by Weather Conditions", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Accidents")
ax.set_ylabel("Weather Condition")
plt.tight_layout()
plt.savefig("viz_1_weather.png", dpi=150)
plt.close()
print("  [+] viz_1_weather.png")

# --- 3b. Accidents by Time of Day ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

time_counts = main_df["Time of Day"].value_counts()
axes[0].bar(
    time_counts.index,
    time_counts.values,
    color=sns.color_palette("viridis", len(time_counts)),
)
axes[0].set_title(
    "Accidents by Time of Day (Main Dataset)", fontsize=12, fontweight="bold"
)
axes[0].set_xlabel("Time of Day")
axes[0].set_ylabel("Number of Accidents")
axes[0].tick_params(axis="x", rotation=45)

time_agg = (
    time_df.groupby("time")["number_of_accidents"].sum().sort_values(ascending=True)
)
axes[1].barh(
    time_agg.index, time_agg.values, color=sns.color_palette("magma", len(time_agg))
)
axes[1].set_title(
    "Accidents by Time Slot (National Data)", fontsize=12, fontweight="bold"
)
axes[1].set_xlabel("Total Accidents")
axes[1].set_ylabel("Time Slot")

plt.tight_layout()
plt.savefig("viz_2_time_of_day.png", dpi=150)
plt.close()
print("  [+] viz_2_time_of_day.png")

# --- 3c. Accidents by Road Type ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

road_counts = main_df["Road Type"].value_counts()
axes[0].barh(
    road_counts.index,
    road_counts.values,
    color=sns.color_palette("Set2", len(road_counts)),
)
axes[0].set_title(
    "Accidents by Road Type (Main Dataset)", fontsize=12, fontweight="bold"
)
axes[0].set_xlabel("Number of Accidents")
axes[0].set_ylabel("Road Type")

road_agg = road_class_df.groupby("road_type")["cases"].sum().sort_values(ascending=True)
axes[1].barh(
    road_agg.index, road_agg.values, color=sns.color_palette("muted", len(road_agg))
)
axes[1].set_title(
    "Road Accidents by Classification (National)", fontsize=12, fontweight="bold"
)
axes[1].set_xlabel("Total Cases")
axes[1].set_ylabel("Road Type")

plt.tight_layout()
plt.savefig("viz_3_road_type.png", dpi=150)
plt.close()
print("  [+] viz_3_road_type.png")

# --- 3d. Accidents by Lighting & Surface ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

light_counts = main_df["Lighting Conditions"].value_counts()
axes[0].barh(
    light_counts.index,
    light_counts.values,
    color=sns.color_palette("YlOrRd", len(light_counts)),
)
axes[0].set_title("Accidents by Lighting Conditions", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Number of Accidents")
axes[0].set_ylabel("Lighting Condition")

surface_counts = main_df["Road Condition"].value_counts()
axes[1].barh(
    surface_counts.index,
    surface_counts.values,
    color=sns.color_palette("Blues", len(surface_counts)),
)
axes[1].set_title("Accidents by Road Surface Condition", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Number of Accidents")
axes[1].set_ylabel("Road Condition")

plt.tight_layout()
plt.savefig("viz_4_lighting_surface.png", dpi=150)
plt.close()
print("  [+] viz_4_lighting_surface.png")

# --- 3e. Place of Occurrence & Monthly Trends ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

place_agg = (
    place_df.groupby("place")["number_of_deaths"]
    .sum()
    .sort_values(ascending=True)
    .tail(10)
)
axes[0].barh(
    place_agg.index, place_agg.values, color=sns.color_palette("Reds", len(place_agg))
)
axes[0].set_title("Top 10 Places of Accident Deaths", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Total Deaths")
axes[0].set_ylabel("Place")

month_order = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
month_agg = month_df.groupby("month")["number_of_accidents"].sum().reindex(month_order)
axes[1].plot(
    month_agg.index, month_agg.values, marker="o", color="crimson", linewidth=2
)
axes[1].set_title(
    "Monthly Trend of Traffic Accidents (National)", fontsize=12, fontweight="bold"
)
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Total Accidents")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("viz_5_place_monthly.png", dpi=150)
plt.close()
print("  [+] viz_5_place_monthly.png")

# ================================================================
# STEP 4: TRANSPORT-WISE SEGREGATED ANALYSIS
# ================================================================
print("\n[4/6] Generating transport-wise analysis...")

# --- 4a. Deaths by Transport Mode ---
t1 = transport_2018_df.groupby("mode_of_transport")["victims"].sum().reset_index()
t1.columns = ["Transport Mode", "Deaths (2018-2020)"]
t2 = transport_2021_df.groupby("mode_of_transport")["died"].sum().reset_index()
t2.columns = ["Transport Mode", "Deaths (2021+)"]

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

t1_sorted = t1.sort_values("Deaths (2018-2020)", ascending=True)
axes[0].barh(
    t1_sorted["Transport Mode"],
    t1_sorted["Deaths (2018-2020)"],
    color=sns.color_palette("rocket", len(t1_sorted)),
)
axes[0].set_title(
    "Deaths by Transport Mode (2018-2020)", fontsize=13, fontweight="bold"
)
axes[0].set_xlabel("Total Deaths")

t2_sorted = t2.sort_values("Deaths (2021+)", ascending=True)
axes[1].barh(
    t2_sorted["Transport Mode"],
    t2_sorted["Deaths (2021+)"],
    color=sns.color_palette("mako", len(t2_sorted)),
)
axes[1].set_title("Deaths by Transport Mode (2021+)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Total Deaths")

plt.suptitle("TRANSPORT-WISE DEATH ANALYSIS", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("analysis_1_transport_deaths.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [+] analysis_1_transport_deaths.png")

# --- 4b. Cause-wise Breakdown ---
road_causes = road_cause_df[
    ~road_cause_df["cause"].isin(["Total Road Accidents", "Weather Condition (Total)"])
]
cause_agg = (
    road_causes.groupby("cause")[["cases", "injured", "died"]]
    .sum()
    .sort_values("cases", ascending=True)
)

fig, ax = plt.subplots(figsize=(14, 8))
y_pos = range(len(cause_agg))
bar_height = 0.25
ax.barh(
    [y - bar_height for y in y_pos],
    cause_agg["cases"],
    height=bar_height,
    label="Cases",
    color="#3498db",
)
ax.barh(
    y_pos, cause_agg["injured"], height=bar_height, label="Injured", color="#f39c12"
)
ax.barh(
    [y + bar_height for y in y_pos],
    cause_agg["died"],
    height=bar_height,
    label="Died",
    color="#e74c3c",
)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(cause_agg.index, fontsize=9)
ax.set_title(
    "CAUSE-WISE DISTRIBUTION: Cases, Injuries & Deaths", fontsize=14, fontweight="bold"
)
ax.set_xlabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig("analysis_2_cause_breakdown.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [+] analysis_2_cause_breakdown.png")

# --- 4c. Vehicle Type vs Severity ---
vehicle_severity = (
    main_df.groupby(["Vehicle Type Involved", "Accident Severity"])
    .size()
    .unstack(fill_value=0)
)
vehicle_severity = vehicle_severity[["Minor", "Serious", "Fatal"]]

fig, ax = plt.subplots(figsize=(12, 7))
vehicle_severity.plot(
    kind="barh", stacked=True, ax=ax, color=["#2ecc71", "#f39c12", "#e74c3c"]
)
ax.set_title("VEHICLE TYPE vs ACCIDENT SEVERITY", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Accidents")
ax.set_ylabel("Vehicle Type")
ax.legend(title="Severity")
plt.tight_layout()
plt.savefig("analysis_3_vehicle_severity.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [+] analysis_3_vehicle_severity.png")

# ================================================================
# STEP 5: COMBINATION ANALYSIS (What Increases / Reduces Accidents)
# ================================================================
print("\n[5/6] Analyzing dangerous & safe combinations...")

combo_df = (
    main_df.groupby(
        [
            "Vehicle Type Involved",
            "Weather Conditions",
            "Road Type",
            "Lighting Conditions",
            "Road Condition",
        ]
    )
    .agg(
        total_accidents=("Accident Severity", "count"),
        fatal_count=("Accident Severity", lambda x: (x == "Fatal").sum()),
        serious_count=("Accident Severity", lambda x: (x == "Serious").sum()),
        avg_casualties=("Number of Casualties", "mean"),
        avg_fatalities=("Number of Fatalities", "mean"),
    )
    .reset_index()
)

combo_df["fatal_rate"] = combo_df["fatal_count"] / combo_df["total_accidents"]
combo_df["severe_rate"] = (
    combo_df["fatal_count"] + combo_df["serious_count"]
) / combo_df["total_accidents"]
combo_significant = combo_df[combo_df["total_accidents"] >= 3].copy()

# --- TOP 15 MOST DANGEROUS ---
dangerous = combo_significant.sort_values("fatal_rate", ascending=False).head(15)
dangerous["combo_label"] = (
    dangerous["Vehicle Type Involved"]
    + " | "
    + dangerous["Weather Conditions"]
    + " | "
    + dangerous["Road Type"]
    + " | "
    + dangerous["Lighting Conditions"]
    + " | "
    + dangerous["Road Condition"]
)

fig, ax = plt.subplots(figsize=(16, 10))
colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(dangerous)))
ax.barh(dangerous["combo_label"], dangerous["fatal_rate"], color=colors)
ax.set_title(
    "TOP 15 MOST DANGEROUS COMBINATIONS (Highest Fatal Rate)",
    fontsize=14,
    fontweight="bold",
)
ax.set_xlabel("Fatal Accident Rate")
for i, (rate, total) in enumerate(
    zip(dangerous["fatal_rate"], dangerous["total_accidents"])
):
    ax.text(rate + 0.01, i, f"{rate:.0%} (n={total})", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("analysis_4_dangerous_combos.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [+] analysis_4_dangerous_combos.png")

# --- TOP 15 SAFEST ---
safe = combo_significant.sort_values("fatal_rate", ascending=True).head(15)
safe["combo_label"] = (
    safe["Vehicle Type Involved"]
    + " | "
    + safe["Weather Conditions"]
    + " | "
    + safe["Road Type"]
    + " | "
    + safe["Lighting Conditions"]
    + " | "
    + safe["Road Condition"]
)

fig, ax = plt.subplots(figsize=(16, 10))
colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(safe)))
ax.barh(safe["combo_label"], safe["fatal_rate"], color=colors)
ax.set_title(
    "TOP 15 SAFEST COMBINATIONS (Lowest Fatal Rate)", fontsize=14, fontweight="bold"
)
ax.set_xlabel("Fatal Accident Rate")
for i, (rate, total) in enumerate(zip(safe["fatal_rate"], safe["total_accidents"])):
    ax.text(rate + 0.005, i, f"{rate:.0%} (n={total})", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("analysis_5_safe_combos.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [+] analysis_5_safe_combos.png")

# ================================================================
# STEP 6: HEATMAPS & DEEP INSIGHTS
# ================================================================
print("\n[6/6] Generating heatmaps & deep insights...")

# --- 6a. Vehicle x Weather Heatmap ---
pivot_weather = main_df.pivot_table(
    index="Vehicle Type Involved",
    columns="Weather Conditions",
    values="Severity_Score",
    aggfunc="mean",
)
fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(
    pivot_weather,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn_r",
    ax=ax,
    linewidths=0.5,
    vmin=1,
    vmax=3,
)
ax.set_title(
    "HEATMAP: Severity by Vehicle Type & Weather\n(1=Minor, 2=Serious, 3=Fatal)",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("analysis_6_heatmap_weather.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [+] analysis_6_heatmap_weather.png")

# --- 6b. Vehicle x Road Condition Heatmap ---
pivot_road = main_df.pivot_table(
    index="Vehicle Type Involved",
    columns="Road Condition",
    values="Severity_Score",
    aggfunc="mean",
)
fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(
    pivot_road,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn_r",
    ax=ax,
    linewidths=0.5,
    vmin=1,
    vmax=3,
)
ax.set_title(
    "HEATMAP: Severity by Vehicle Type & Road Condition\n(1=Minor, 2=Serious, 3=Fatal)",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("analysis_7_heatmap_road.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [+] analysis_7_heatmap_road.png")

# --- 6c. Vehicle x Lighting Heatmap ---
pivot_light = main_df.pivot_table(
    index="Vehicle Type Involved",
    columns="Lighting Conditions",
    values="Severity_Score",
    aggfunc="mean",
)
fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(
    pivot_light,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn_r",
    ax=ax,
    linewidths=0.5,
    vmin=1,
    vmax=3,
)
ax.set_title(
    "HEATMAP: Severity by Vehicle Type & Lighting\n(1=Minor, 2=Serious, 3=Fatal)",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("analysis_8_heatmap_lighting.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [+] analysis_8_heatmap_lighting.png")

# --- 6d. Alcohol & Speed Impact ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

alcohol_rate = (
    main_df.groupby("Vehicle Type Involved")["Alcohol Involvement"]
    .apply(lambda x: (x == "Yes").sum() / len(x))
    .sort_values()
)
axes[0].barh(
    alcohol_rate.index,
    alcohol_rate.values,
    color=sns.color_palette("flare", len(alcohol_rate)),
)
axes[0].set_title(
    "Alcohol Involvement Rate by Vehicle Type", fontsize=12, fontweight="bold"
)
axes[0].set_xlabel("Rate")
for i, v in enumerate(alcohol_rate.values):
    axes[0].text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=9)

speed_by_severity = (
    main_df.groupby(["Vehicle Type Involved", "Accident Severity"])[
        "Speed Limit (km/h)"
    ]
    .mean()
    .unstack()
)
speed_by_severity[["Minor", "Serious", "Fatal"]].plot(
    kind="barh", ax=axes[1], color=["#2ecc71", "#f39c12", "#e74c3c"]
)
axes[1].set_title(
    "Avg Speed Limit by Vehicle Type & Severity", fontsize=12, fontweight="bold"
)
axes[1].set_xlabel("Avg Speed Limit (km/h)")
axes[1].legend(title="Severity")

plt.tight_layout()
plt.savefig("analysis_9_alcohol_speed.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [+] analysis_9_alcohol_speed.png")

# ================================================================
# FINAL SUMMARY REPORT
# ================================================================
print("\n" + "=" * 70)
print("  COMPLETE ANALYSIS REPORT")
print("=" * 70)

print(
    f"\n  Dataset: {main_df.shape[0]} accidents across {main_df['State Name'].nunique()} states"
)
print(f"  Vehicle Types: {', '.join(main_df['Vehicle Type Involved'].unique())}")

print("\n" + "-" * 50)
print("  MOST DANGEROUS COMBINATIONS (100% Fatal Rate)")
print("-" * 50)
for _, row in dangerous.head(5).iterrows():
    print(f"  >> {row['combo_label']}")
    print(
        f"     Fatal Rate: {row['fatal_rate']:.0%} | Accidents: {row['total_accidents']} | Avg Fatalities: {row['avg_fatalities']:.1f}"
    )

print("\n" + "-" * 50)
print("  SAFEST COMBINATIONS (Lowest Fatal Rate)")
print("-" * 50)
for _, row in safe.head(5).iterrows():
    print(f"  >> {row['combo_label']}")
    print(
        f"     Fatal Rate: {row['fatal_rate']:.0%} | Accidents: {row['total_accidents']} | Avg Casualties: {row['avg_casualties']:.1f}"
    )

print("\n" + "-" * 50)
print("  TOP CAUSES OF ROAD ACCIDENTS (National)")
print("-" * 50)
cause_top = (
    road_causes.groupby("cause")["cases"].sum().sort_values(ascending=False).head(5)
)
for cause, count in cause_top.items():
    print(f"  {cause}: {count:,.0f} cases")

print("\n" + "-" * 50)
print("  DEADLIEST TRANSPORT MODES (2021+)")
print("-" * 50)
for _, row in t2_sorted.tail(5).iloc[::-1].iterrows():
    print(f"  {row['Transport Mode']}: {row['Deaths (2021+)']:,.0f} deaths")

print("\n" + "-" * 50)
print("  KEY INSIGHTS")
print("-" * 50)
veh_fatal = (
    main_df.groupby("Vehicle Type Involved")["Severity_Score"]
    .mean()
    .sort_values(ascending=False)
)
print(
    f"  Most dangerous vehicle:  {veh_fatal.index[0]} (severity: {veh_fatal.iloc[0]:.2f}/3)"
)
print(
    f"  Safest vehicle:          {veh_fatal.index[-1]} (severity: {veh_fatal.iloc[-1]:.2f}/3)"
)

weather_fatal = (
    main_df.groupby("Weather Conditions")["Severity_Score"]
    .mean()
    .sort_values(ascending=False)
)
print(
    f"  Most dangerous weather:  {weather_fatal.index[0]} (severity: {weather_fatal.iloc[0]:.2f}/3)"
)
print(
    f"  Safest weather:          {weather_fatal.index[-1]} (severity: {weather_fatal.iloc[-1]:.2f}/3)"
)

road_fatal = (
    main_df.groupby("Road Condition")["Severity_Score"]
    .mean()
    .sort_values(ascending=False)
)
print(
    f"  Most dangerous road:     {road_fatal.index[0]} (severity: {road_fatal.iloc[0]:.2f}/3)"
)
print(
    f"  Safest road:             {road_fatal.index[-1]} (severity: {road_fatal.iloc[-1]:.2f}/3)"
)

light_fatal = (
    main_df.groupby("Lighting Conditions")["Severity_Score"]
    .mean()
    .sort_values(ascending=False)
)
print(
    f"  Most dangerous lighting: {light_fatal.index[0]} (severity: {light_fatal.iloc[0]:.2f}/3)"
)
print(
    f"  Safest lighting:         {light_fatal.index[-1]} (severity: {light_fatal.iloc[-1]:.2f}/3)"
)

print("\n" + "=" * 70)
print("  ALL 14 CHARTS GENERATED SUCCESSFULLY")
print("=" * 70)
print("  Basic Visualizations:")
print("    viz_1_weather.png          - Accidents by weather")
print("    viz_2_time_of_day.png      - Accidents by time of day")
print("    viz_3_road_type.png        - Accidents by road type")
print("    viz_4_lighting_surface.png - Lighting & surface conditions")
print("    viz_5_place_monthly.png    - Place of death & monthly trends")
print("  Transport & Cause Analysis:")
print("    analysis_1_transport_deaths.png  - Deaths by transport mode")
print("    analysis_2_cause_breakdown.png   - Cause-wise breakdown")
print("    analysis_3_vehicle_severity.png  - Vehicle type vs severity")
print("    analysis_4_dangerous_combos.png  - Top 15 dangerous combos")
print("    analysis_5_safe_combos.png       - Top 15 safe combos")
print("  Heatmaps & Deep Insights:")
print("    analysis_6_heatmap_weather.png   - Vehicle x Weather severity")
print("    analysis_7_heatmap_road.png      - Vehicle x Road severity")
print("    analysis_8_heatmap_lighting.png  - Vehicle x Lighting severity")
print("    analysis_9_alcohol_speed.png     - Alcohol & speed impact")
print("=" * 70)
