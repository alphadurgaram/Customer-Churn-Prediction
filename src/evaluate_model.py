
# ==========================================
# CUSTOMER CHURN PREDICTION
# Step 5: Model Evaluation
# ==========================================

import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# 1. Load Dataset and Trained Model
df = pd.read_csv("data/processed/cleaned_churn_data.csv")
model = joblib.load("models/churn_model.pkl")

print("Dataset Loaded:", df.shape)
print("Trained Model Loaded Successfully!")

# 2. Prepare Features and Target
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# 3. Split Dataset
# Keep the same split settings used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Make Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 5. Calculate Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n========== MODEL PERFORMANCE ==========")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

# 6. Classification Report
print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(
    y_test,
    y_pred,
    target_names=["No Churn", "Churn"]
))

# 7. Create Figures Directory
os.makedirs("reports/figures", exist_ok=True)

# 8. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)
plt.title("Customer Churn - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("reports/figures/confusion_matrix.png")
plt.show()
plt.close()

# 9. ROC Curve
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("Customer Churn - ROC Curve")
plt.tight_layout()
plt.savefig("reports/figures/roc_curve.png")
plt.show()
plt.close()

print("\nEvaluation Completed Successfully!")
print("Evaluation graphs saved in reports/figures/")
