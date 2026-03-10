import pandas as pd

df = pd.read_csv("accident_prediction_india.csv")

print(f"Number of rows: {len(df)}")
null = df.isnull().sum()
print(f"Number of null values: {null.sum()}")

print("\nData Types:")
print(df.dtypes)

# Binary indicator columns for missingness
df["Traffic_Control_Missing"] = df["Traffic Control Presence"].isnull().astype(int)
df["License_Status_Missing"] = df["Driver License Status"].isnull().astype(int)

# Fill missing values with "Unknown"
df["Traffic Control Presence"].fillna("Unknown", inplace=True)
df["Driver License Status"].fillna("Unknown", inplace=True)

print(f"\nNull values after filling: {df.isnull().sum().sum()}")
print(f"New columns added: Traffic_Control_Missing, License_Status_Missing")
print(f"Total columns: {len(df.columns)}")
