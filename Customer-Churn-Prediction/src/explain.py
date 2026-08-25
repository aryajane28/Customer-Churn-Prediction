"""
SHAP explainability for the trained Random Forest pipeline.

Run:
    python src/explain.py

If SHAP is unavailable or incompatible with the installed environment,
the script prints a helpful message instead of failing silently.
"""
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "customer_churn.csv"
MODEL_PATH = ROOT / "models" / "random_forest.joblib"
REPORTS = ROOT / "reports"


def main():
    try:
        import shap
    except ImportError:
        print("SHAP is not installed. Run: pip install shap")
        return

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["churn", "customer_id"]).head(300)

    # Transform through the fitted preprocessing pipeline.
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    X_transformed = preprocessor.transform(X)

    feature_names = preprocessor.get_feature_names_out()

    # TreeExplainer works directly with the transformed Random Forest matrix.
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_transformed)

    # Binary classification can return a list in some SHAP versions.
    values = shap_values[1] if isinstance(shap_values, list) else shap_values
    if getattr(values, "ndim", 2) == 3:
        values = values[:, :, 1]

    plt.figure()
    shap.summary_plot(
        values,
        X_transformed,
        feature_names=feature_names,
        show=False,
        max_display=15
    )
    plt.tight_layout()
    plt.savefig(REPORTS / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()

    print("Saved SHAP summary to reports/shap_summary.png")


if __name__ == "__main__":
    main()
