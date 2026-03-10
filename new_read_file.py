import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# --- Load all CSV files ---
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

# ============================================================
# 1. ACCIDENTS BY WEATHER (from main dataset)
# ============================================================
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
plt.savefig("viz_accidents_by_weather.png", dpi=150)
plt.close()

# ============================================================
# 2. ACCIDENTS BY TIME OF DAY
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 2a. From main dataset
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

# 2b. From time-of-occurrence dataset (aggregated by time slot)
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
plt.savefig("viz_accidents_by_time.png", dpi=150)
plt.close()

# ============================================================
# 3. ACCIDENTS BY ROAD TYPE / JUNCTION TYPE
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 3a. From main dataset - Road Type
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

# 3b. From road-classification dataset (aggregated)
road_agg = road_class_df.groupby("road_type")["cases"].sum().sort_values(ascending=True)
axes[1].barh(
    road_agg.index, road_agg.values, color=sns.color_palette("muted", len(road_agg))
)
axes[1].set_title(
    "Road Accidents by Road Classification (National Data)",
    fontsize=12,
    fontweight="bold",
)
axes[1].set_xlabel("Total Cases")
axes[1].set_ylabel("Road Type")

plt.tight_layout()
plt.savefig("viz_accidents_by_road_type.png", dpi=150)
plt.close()

# ============================================================
# 4. ACCIDENTS BY LIGHTING / SURFACE CONDITIONS
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 4a. Lighting Conditions (from main dataset)
light_counts = main_df["Lighting Conditions"].value_counts()
axes[0].barh(
    light_counts.index,
    light_counts.values,
    color=sns.color_palette("YlOrRd", len(light_counts)),
)
axes[0].set_title("Accidents by Lighting Conditions", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Number of Accidents")
axes[0].set_ylabel("Lighting Condition")

# 4b. Road Surface / Condition (from main dataset)
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
plt.savefig("viz_accidents_by_light_surface.png", dpi=150)
plt.close()

# ============================================================
# 5. BONUS: Place of Occurrence & Monthly Trends
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 5a. Place of occurrence (deaths)
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

# 5b. Monthly trend of accidents
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
month_agg = month_df.groupby("month")["number_of_accidents"].sum()
month_agg = month_agg.reindex(month_order)
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
plt.savefig("viz_place_and_monthly_trend.png", dpi=150)
plt.close()

print("All visualizations saved as PNG files:")
print("  - viz_accidents_by_weather.png")
print("  - viz_accidents_by_time.png")
print("  - viz_accidents_by_road_type.png")
print("  - viz_accidents_by_light_surface.png")
print("  - viz_place_and_monthly_trend.png")
