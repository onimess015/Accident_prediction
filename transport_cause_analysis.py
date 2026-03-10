import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")

# --- Load all datasets ---
main_df = pd.read_csv("accident_prediction_india.csv")
road_cause_df = pd.read_csv(
    "cause-wise-distribution-of-road-accidents-and-unmanned-railway-crossing-accidents.csv"
)
railway_cause_df = pd.read_csv("cause-wise-distribution-of-railway-accidents.csv")
transport_2018_df = pd.read_csv(
    "mode-of-transport-wise-number-of-persons-died-in-road-accidents-2018-2020.csv"
)
transport_2021_df = pd.read_csv(
    "mode-of-transport-wise-number-of-persons-died-in-road-accidents-onwards-2021.csv"
)
cases_df = pd.read_csv("cases-reported-persons-injured-and-died.csv")

# ============================================================
# SECTION 1: TRANSPORT-WISE DEATH ANALYSIS
# ============================================================

# Combine transport data (2018-2020 has offenders/victims, 2021+ has injured/died)
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

# ============================================================
# SECTION 2: CAUSE-WISE BREAKDOWN FOR ROAD ACCIDENTS
# ============================================================
# Exclude totals row
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

ax.set_yticks(y_pos)
ax.set_yticklabels(cause_agg.index, fontsize=9)
ax.set_title(
    "CAUSE-WISE DISTRIBUTION: Cases, Injuries & Deaths", fontsize=14, fontweight="bold"
)
ax.set_xlabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig("analysis_2_cause_breakdown.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# SECTION 3: VEHICLE TYPE vs SEVERITY (Main Dataset)
# ============================================================
severity_map = {"Minor": 1, "Serious": 2, "Fatal": 3}
main_df["Severity_Score"] = main_df["Accident Severity"].map(severity_map)

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

# ============================================================
# SECTION 4: COMBINATIONS THAT INCREASE ACCIDENTS (High Risk)
# ============================================================
# Find combinations with highest fatal accident rates
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

# Filter: only combos with enough data points
combo_significant = combo_df[combo_df["total_accidents"] >= 3].copy()

# --- TOP 15 MOST DANGEROUS COMBINATIONS ---
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
plt.savefig("analysis_4_dangerous_combinations.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# SECTION 5: COMBINATIONS THAT REDUCE ACCIDENTS (Low Risk)
# ============================================================
# --- TOP 15 SAFEST COMBINATIONS ---
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
plt.savefig("analysis_5_safe_combinations.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# SECTION 6: HEATMAP - Vehicle Type vs Weather (Fatality Rate)
# ============================================================
pivot_fatal = main_df.pivot_table(
    index="Vehicle Type Involved",
    columns="Weather Conditions",
    values="Severity_Score",
    aggfunc="mean",
)

fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(
    pivot_fatal,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn_r",
    ax=ax,
    linewidths=0.5,
    vmin=1,
    vmax=3,
)
ax.set_title(
    "HEATMAP: Avg Severity by Vehicle Type & Weather\n(1=Minor, 2=Serious, 3=Fatal)",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("analysis_6_heatmap_vehicle_weather.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# SECTION 7: HEATMAP - Vehicle Type vs Road Condition (Fatality Rate)
# ============================================================
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
    "HEATMAP: Avg Severity by Vehicle Type & Road Condition\n(1=Minor, 2=Serious, 3=Fatal)",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("analysis_7_heatmap_vehicle_road.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# SECTION 8: HEATMAP - Vehicle Type vs Lighting (Fatality Rate)
# ============================================================
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
    "HEATMAP: Avg Severity by Vehicle Type & Lighting\n(1=Minor, 2=Serious, 3=Fatal)",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("analysis_8_heatmap_vehicle_lighting.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# SECTION 9: ALCOHOL & SPEED IMPACT PER VEHICLE TYPE
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Alcohol involvement rate per vehicle type
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

# Avg speed limit per vehicle type (in fatal vs non-fatal)
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

# ============================================================
# PRINT SUMMARY REPORT
# ============================================================
print("=" * 70)
print("TRANSPORT & CAUSE ANALYSIS - SUMMARY REPORT")
print("=" * 70)

print("\n--- MOST DANGEROUS COMBINATIONS (Fatal Rate) ---")
for _, row in dangerous.head(5).iterrows():
    print(f"  {row['combo_label']}")
    print(
        f"    Fatal Rate: {row['fatal_rate']:.0%} | Accidents: {row['total_accidents']} | Avg Fatalities: {row['avg_fatalities']:.1f}"
    )

print("\n--- SAFEST COMBINATIONS (Low Fatal Rate) ---")
for _, row in safe.head(5).iterrows():
    print(f"  {row['combo_label']}")
    print(
        f"    Fatal Rate: {row['fatal_rate']:.0%} | Accidents: {row['total_accidents']} | Avg Casualties: {row['avg_casualties']:.1f}"
    )

print("\n--- TOP CAUSES OF ROAD ACCIDENTS (National) ---")
cause_top = (
    road_causes.groupby("cause")["cases"].sum().sort_values(ascending=False).head(5)
)
for cause, count in cause_top.items():
    print(f"  {cause}: {count:,.0f} cases")

print("\n--- DEADLIEST TRANSPORT MODES (2021+) ---")
for _, row in t2_sorted.tail(5).iloc[::-1].iterrows():
    print(f"  {row['Transport Mode']}: {row['Deaths (2021+)']:,.0f} deaths")

print("\n--- KEY INSIGHTS ---")
# Vehicle with highest fatal rate
veh_fatal = (
    main_df.groupby("Vehicle Type Involved")["Severity_Score"]
    .mean()
    .sort_values(ascending=False)
)
print(
    f"  Highest avg severity vehicle: {veh_fatal.index[0]} (score: {veh_fatal.iloc[0]:.2f}/3)"
)
print(
    f"  Lowest avg severity vehicle:  {veh_fatal.index[-1]} (score: {veh_fatal.iloc[-1]:.2f}/3)"
)

# Weather with highest fatal rate
weather_fatal = (
    main_df.groupby("Weather Conditions")["Severity_Score"]
    .mean()
    .sort_values(ascending=False)
)
print(
    f"  Most dangerous weather: {weather_fatal.index[0]} (score: {weather_fatal.iloc[0]:.2f}/3)"
)
print(
    f"  Least dangerous weather: {weather_fatal.index[-1]} (score: {weather_fatal.iloc[-1]:.2f}/3)"
)

print("\nAll analysis charts saved as PNG files.")
print("  analysis_1_transport_deaths.png      - Deaths by transport mode")
print("  analysis_2_cause_breakdown.png       - Cause-wise cases/injuries/deaths")
print("  analysis_3_vehicle_severity.png      - Vehicle type vs severity")
print("  analysis_4_dangerous_combinations.png - Top 15 dangerous combos")
print("  analysis_5_safe_combinations.png     - Top 15 safe combos")
print("  analysis_6_heatmap_vehicle_weather.png - Vehicle x Weather heatmap")
print("  analysis_7_heatmap_vehicle_road.png  - Vehicle x Road heatmap")
print("  analysis_8_heatmap_vehicle_lighting.png - Vehicle x Lighting heatmap")
print("  analysis_9_alcohol_speed.png         - Alcohol & speed impact")
