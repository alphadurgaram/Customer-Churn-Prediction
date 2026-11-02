
# ==========================================
# CUSTOMER CHURN PREDICTION
# Step 4: Machine Learning Model Training
# ==========================================

import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# 1. Load Cleaned Dataset
df = pd.read_csv("data/processed/cleaned_churn_data.csv")

print("Dataset Loaded:", df.shape)

# 2. Remove Customer ID
df = df.drop(columns=["customerID"])

# 3. Separate Features and Target
X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# 4. Identify Numerical and Categorical Columns
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "str"]).columns.tolist()

print("\nNumerical Features:", numerical_features)
print("Categorical Features:", categorical_features)

# 5. Preprocessing Pipelines
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 6. Complete Machine Learning Pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

# 7. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", X_train.shape[0])
print("Testing Records:", X_test.shape[0])

# 8. Train Model
model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")

# 9. Save Trained Model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/churn_model.pkl")

print("Model Saved: models/churn_model.pkl")
