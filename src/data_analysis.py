
# ==========================================
# CUSTOMER CHURN PREDICTION
# Step 1: Basic Data Analysis
# ==========================================

import pandas as pd

# 1. Load Dataset
DATA_PATH = "data/raw/Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

# 2. Dataset Overview
print("\n========== DATASET OVERVIEW ==========")
print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

# 3. Display First 5 Records
print("\n========== FIRST 5 RECORDS ==========")
print(df.head())

# 4. Dataset Information
print("\n========== DATASET INFORMATION ==========")
df.info()

# 5. Check Missing Values
print("\n========== MISSING VALUES ==========")

# Detect empty strings and whitespace as missing
df.replace(r"^\s*$", pd.NA, regex=True, inplace=True)

print(df.isnull().sum())

# 6. Check Duplicate Records
print("\n========== DUPLICATE RECORDS ==========")
print("Total Duplicates:", df.duplicated().sum())

# 7. Churn Distribution
print("\n========== CHURN DISTRIBUTION ==========")
print(df["Churn"].value_counts())

# 8. Churn Percentage
print("\n========== CHURN PERCENTAGE ==========")
churn_percentage = df["Churn"].value_counts(normalize=True) * 100
print(churn_percentage.round(2))

# 9. Numerical Data Summary
print("\n========== NUMERICAL DATA SUMMARY ==========")
print(df.describe())

# 10. Categorical Data Summary
print("\n========== CATEGORICAL DATA SUMMARY ==========")
print(df.describe(include="object"))

print("\n========== ANALYSIS COMPLETED ==========")
