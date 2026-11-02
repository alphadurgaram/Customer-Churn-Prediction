
# ==========================================
# CUSTOMER CHURN PREDICTION
# Step 2: Data Preprocessing
# ==========================================

import pandas as pd

# Load original dataset
DATA_PATH = "data/raw/Telco-Customer-Churn.csv"
OUTPUT_PATH = "data/processed/cleaned_churn_data.csv"

df = pd.read_csv(DATA_PATH)

print("Original Dataset Shape:", df.shape)

# 1. Convert TotalCharges into numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# 2. Check missing values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# 3. Remove rows with missing TotalCharges
df.dropna(subset=["TotalCharges"], inplace=True)

# 4. Remove duplicate records
df.drop_duplicates(inplace=True)

# 5. Reset index
df.reset_index(drop=True, inplace=True)

# 6. Verify cleaned dataset
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nCleaned Dataset Shape:", df.shape)

# 7. Save cleaned dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaned dataset saved successfully!")
print("File Location:", OUTPUT_PATH)
