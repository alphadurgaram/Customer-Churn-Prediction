
# ==========================================
# CUSTOMER CHURN PREDICTION
# Step 3: Exploratory Data Analysis
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_churn_data.csv")

# Create figures directory
os.makedirs("reports/figures", exist_ok=True)

sns.set_theme(style="whitegrid")

# 1. Customer Churn Distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.savefig("reports/figures/churn_distribution.png")
plt.show()
plt.close()

# 2. Gender vs Churn
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="gender", hue="Churn")
plt.title("Gender vs Customer Churn")
plt.savefig("reports/figures/gender_vs_churn.png")
plt.show()
plt.close()

# 3. Contract Type vs Churn
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Contract Type vs Customer Churn")
plt.xticks(rotation=15)
plt.savefig("reports/figures/contract_vs_churn.png")
plt.show()
plt.close()

# 4. Tenure vs Churn
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="tenure")
plt.title("Tenure vs Customer Churn")
plt.savefig("reports/figures/tenure_vs_churn.png")
plt.show()
plt.close()

# 5. Monthly Charges vs Churn
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges vs Customer Churn")
plt.savefig("reports/figures/monthly_charges_vs_churn.png")
plt.show()
plt.close()

# 6. Payment Method vs Churn
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="PaymentMethod", hue="Churn")
plt.title("Payment Method vs Customer Churn")
plt.xticks(rotation=20)
plt.savefig("reports/figures/payment_method_vs_churn.png")
plt.show()
plt.close()

print("\nEDA COMPLETED SUCCESSFULLY!")
print("All graphs saved in reports/figures/")
