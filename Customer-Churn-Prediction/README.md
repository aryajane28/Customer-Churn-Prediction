# Customer Churn Prediction

An end-to-end machine learning project for predicting customer churn using **Python, Pandas, Scikit-learn, Random Forest, Logistic Regression, and SHAP**.

## Project Highlights

- Built a reproducible customer churn classification pipeline.
- Performed data cleaning and missing-value handling.
- Applied one-hot encoding to categorical variables.
- Applied feature scaling to numerical variables.
- Trained and compared Logistic Regression and Random Forest classifiers.
- Evaluated models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- Added confusion-matrix and ROC-curve visualizations.
- Added SHAP-based model explainability.
- Uses `random_state=42` for reproducibility.
- Includes a synthetic dataset so the repository can be run without external data access.

> **Important:** The performance metrics in `reports/metrics.json` are generated from the included synthetic dataset. Run `python src/train.py` to regenerate and verify them.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- SHAP

## Repository Structure

```text
Customer-Churn-Prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── customer_churn.csv
│   └── README.md
├── models/
│   └── .gitkeep
├── notebooks/
│   └── README.md
├── reports/
│   ├── metrics.json
│   ├── confusion_matrix.png
│   └── roc_curve.png
└── src/
    ├── train.py
    └── explain.py
```

## Dataset

The repository contains a reproducible synthetic customer dataset with features such as:

- Age
- Tenure
- Monthly charges
- Total charges
- Support calls
- Late payments
- Satisfaction score
- Number of products
- Contract type
- Internet service
- Payment method
- Paperless billing
- Churn target

The dataset is intentionally synthetic and is included for demonstration and portfolio reproducibility.

## Machine Learning Workflow

1. Load customer data.
2. Remove the customer identifier from model features.
3. Separate features and churn target.
4. Split data into training and test sets using stratification.
5. Impute missing numerical and categorical values.
6. One-hot encode categorical variables.
7. Scale numerical variables.
8. Train Logistic Regression.
9. Train Random Forest.
10. Compare classification metrics.
11. Save trained models with Joblib.
12. Generate confusion matrix and ROC curve.
13. Generate SHAP explanations for the Random Forest model.

## Installation

```bash
git clone https://github.com/aryajane28/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Train the Models

From the project root:

```bash
python src/train.py
```

This creates:

```text
models/logistic_regression.joblib
models/random_forest.joblib
reports/metrics.json
reports/confusion_matrix.png
reports/roc_curve.png
```

## Generate SHAP Explainability

After training:

```bash
python src/explain.py
```

Output:

```text
reports/shap_summary.png
```

## Evaluation Metrics

The project evaluates:

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| Precision | Percentage of predicted churners who actually churn |
| Recall | Percentage of actual churners correctly identified |
| F1-Score | Balance between precision and recall |
| ROC-AUC | Ranking/discrimination performance across thresholds |

The exact metrics are stored in `reports/metrics.json` and can be regenerated with:

```bash
python src/train.py
```

## Why Random Forest?

Random Forest is useful for customer churn because it can capture nonlinear relationships and interactions between customer characteristics without requiring strong linear assumptions.

Logistic Regression is included as a transparent baseline model.

## Explainability

SHAP is used to understand which transformed features have the greatest influence on Random Forest churn predictions. This makes the model easier to interpret and demonstrates an important step toward responsible machine learning.

## Reproducibility

All major randomized operations use:

```python
random_state=42
```

The included dataset is also generated from a fixed seed, making the project reproducible.

## Portfolio Summary

**Customer Churn Prediction | Python, Scikit-learn, Pandas, Random Forest, SHAP**

- Trained and compared Logistic Regression and Random Forest classifiers on customer data to predict churn.
- Built an end-to-end ML pipeline covering data cleaning, encoding, feature scaling, model training, and evaluation.
- Evaluated models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- Added SHAP explainability and model-performance visualizations.
- Designed the project to be reproducible with a fixed random seed and included synthetic dataset.

## License

This project is intended for educational and portfolio use.
