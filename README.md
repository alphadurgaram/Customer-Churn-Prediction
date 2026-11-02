# Customer Churn Prediction using Machine Learning

## Project Overview
This project focuses on predicting customer churn using Machine Learning techniques. It analyzes customer information and service usage patterns to identify customers who are likely to discontinue a company's services.

The project uses the IBM Telco Customer Churn dataset and compares Logistic Regression and Random Forest classification models.

## Objectives
- Analyze customer data and identify churn patterns.
- Perform data cleaning and preprocessing.
- Explore relationships between customer features and churn.
- Train Machine Learning classification models.
- Evaluate model performance using standard metrics.
- Compare models to understand their strengths and limitations.

## Dataset
**Dataset:** IBM Telco Customer Churn

- Original records: 7,043
- Features: 21
- Records after preprocessing: 7,032
- Target variable: `Churn`
- Target classes: Yes and No

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Project Structure
```text
Customer Churn Prediction/
│
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv
│   └── processed/
│       └── cleaned_churn_data.csv
│
├── notebooks/
├── src/
│   ├── data_analysis.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── compare_models.py
│
├── models/
│   ├── churn_model.pkl
│   └── random_forest_model.pkl
│
├── reports/
│   ├── figures/
│   ├── weekly_reports/
│   └── model_comparison.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Project Workflow
1. Data Collection
2. Data Analysis
3. Data Cleaning and Preprocessing
4. Exploratory Data Analysis (EDA)
5. Model Training
6. Model Evaluation
7. Model Comparison

## Model Performance

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 80.38% | 73.92% |
| Precision | 64.85% | 50.62% |
| Recall | 57.22% | 76.20% |
| F1 Score | 60.80% | 60.83% |
| ROC-AUC | 83.59% | 82.98% |

### Model Comparison
Logistic Regression achieved higher accuracy, precision, and ROC-AUC.

Random Forest achieved higher recall, meaning it identified a larger proportion of actual churn customers.

The F1 scores of both models were nearly identical. The choice of model depends on the business objective.

## How to Run the Project

### 1. Clone the repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project directory
```bash
cd "Customer Churn Prediction"
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run data analysis
```bash
python src/data_analysis.py
```

### 5. Run preprocessing
```bash
python src/preprocessing.py
```

### 6. Generate EDA visualizations
```bash
python src/eda.py
```

### 7. Train the model
```bash
python src/train_model.py
```

### 8. Evaluate the model
```bash
python src/evaluate_model.py
```

### 9. Compare models
```bash
python src/compare_models.py
```

## Conclusion
This project demonstrates how Machine Learning can be used to analyze customer behavior and predict potential churn.

Both Logistic Regression and Random Forest were trained and evaluated. Their results highlight the trade-off between precision and recall when identifying customers likely to leave a service.

## Author
B.Tech Computer Science and Engineering Student

## Disclaimer
This project is developed for educational and learning purposes.