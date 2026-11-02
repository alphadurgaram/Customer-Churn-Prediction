
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_churn_data.csv"
LR_MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
RF_MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"
COMPARISON_PATH = BASE_DIR / "reports" / "model_comparison.csv"


# --------------------------------------------------
# LOAD DATA AND MODELS
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


# --------------------------------------------------
# CHECK REQUIRED FILES
# --------------------------------------------------

required_files = [
    DATA_PATH,
    LR_MODEL_PATH,
    RF_MODEL_PATH,
    COMPARISON_PATH
]

missing_files = [str(path) for path in required_files if not path.exists()]

if missing_files:
    st.error("Required project files are missing:")
    for file in missing_files:
        st.write(f"- {file}")
    st.stop()

try:
    df = load_data()
    lr_model = load_model(LR_MODEL_PATH)
    rf_model = load_model(RF_MODEL_PATH)
    comparison_df = pd.read_csv(COMPARISON_PATH)

except Exception as error:
    st.error(f"Could not load project files: {error}")
    st.stop()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Customer Churn AI")
st.sidebar.caption("Machine Learning Prediction System")

page = st.sidebar.radio(
    "Navigation",
    [
        "Single Prediction",
        "Analytics Dashboard",
        "Model Comparison"
    ]
)

st.sidebar.divider()
st.sidebar.info(
    "Built using Python, Streamlit and Scikit-learn."
)


# --------------------------------------------------
# PAGE 1: SINGLE CUSTOMER PREDICTION
# --------------------------------------------------

if page == "Single Prediction":

    st.title("Customer Churn Prediction")
    st.write(
        "Enter customer details to predict whether "
        "the customer is likely to leave the service."
    )

    selected_model = st.selectbox(
        "Select Machine Learning Model",
        ["Logistic Regression", "Random Forest"]
    )

    model = (
        lr_model
        if selected_model == "Logistic Regression"
        else rf_model
    )

    with st.form("customer_form"):

        st.subheader("Customer Information")

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox(
                "Gender", ["Male", "Female"]
            )

            senior_citizen = st.selectbox(
                "Senior Citizen", [0, 1]
            )

            partner = st.selectbox(
                "Partner", ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Dependents", ["Yes", "No"]
            )

            tenure = st.slider(
                "Tenure (Months)", 0, 72, 12
            )

            phone_service = st.selectbox(
                "Phone Service", ["Yes", "No"]
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"]
            )

            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )

        with col2:
            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )

            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )

            paperless_billing = st.selectbox(
                "Paperless Billing", ["Yes", "No"]
            )

            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=2000.0,
                value=70.0
            )

            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                max_value=100000.0,
                value=float(tenure * monthly_charges)
            )

        submitted = st.form_submit_button(
            "Predict Customer Churn",
            use_container_width=True
        )

    if submitted:

        customer_data = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }])

        try:
            prediction = model.predict(customer_data)[0]
            probability = model.predict_proba(customer_data)[0][1]

            st.divider()
            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            with col1:
                if prediction == 1:
                    st.error("High Churn Alert: Customer predicted to leave.")
                else:
                    st.success("Customer predicted to stay.")

            with col2:
                st.metric(
                    "Predicted Churn Probability",
                    f"{probability * 100:.2f}%"
                )

            st.progress(float(probability))

            st.caption(
                "This probability is the model's estimated likelihood "
                "of churn, not a guarantee of customer behavior."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# PAGE 2: ANALYTICS DASHBOARD
# --------------------------------------------------

elif page == "Analytics Dashboard":

    st.title("Customer Analytics Dashboard")
    st.write("Explore customer churn patterns in the dataset.")

    total_customers = len(df)
    churned = (df["Churn"] == "Yes").sum()
    retained = (df["Churn"] == "No").sum()
    churn_rate = (churned / total_customers) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churned Customers", f"{churned:,}")
    col3.metric("Retained Customers", f"{retained:,}")
    col4.metric("Churn Rate", f"{churn_rate:.2f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Churn Distribution")

        churn_counts = (
            df["Churn"]
            .value_counts()
            .reindex(["No", "Yes"], fill_value=0)
        )

        st.bar_chart(churn_counts)

    with right:
        st.subheader("Contract Type vs Churn")

        contract_data = pd.crosstab(
            df["Contract"],
            df["Churn"]
        )

        st.bar_chart(contract_data)

    left, right = st.columns(2)

    with left:
        st.subheader("Average Tenure by Churn")

        tenure_data = df.groupby("Churn")["tenure"].mean()
        st.bar_chart(tenure_data)

    with right:
        st.subheader("Average Monthly Charges by Churn")

        charges_data = df.groupby("Churn")["MonthlyCharges"].mean()
        st.bar_chart(charges_data)

    st.divider()

    st.subheader("Payment Method vs Churn")

    payment_data = pd.crosstab(
        df["PaymentMethod"],
        df["Churn"]
    )

    st.bar_chart(payment_data)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)


# --------------------------------------------------
# PAGE 3: MODEL COMPARISON
# --------------------------------------------------

elif page == "Model Comparison":

    st.title("Machine Learning Model Comparison")

    st.write(
        "Compare Logistic Regression and Random Forest "
        "using the saved evaluation results."
    )

    st.dataframe(
        comparison_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]

    available_metrics = [
        col for col in metric_columns
        if col in comparison_df.columns
    ]

    if available_metrics and "Model" in comparison_df.columns:

        selected_metric = st.selectbox(
            "Select Evaluation Metric",
            available_metrics
        )

        chart_data = comparison_df.set_index("Model")[
            selected_metric
        ]

        st.subheader(f"{selected_metric} Comparison")
        st.bar_chart(chart_data)

    st.info(
        "Logistic Regression has higher accuracy and ROC-AUC. "
        "Random Forest has higher recall. Model selection depends "
        "on the business objective."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()
st.caption(
    "Customer Churn Prediction | Python + Machine Learning + Streamlit"
)
