from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from train_models import calculate_metrics


PROJECT_DIR = Path(__file__).parent
MODEL_DIR = PROJECT_DIR / "model"
TARGET_COLUMN = "diagnosis"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k-nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Classification Lab", page_icon="ML", layout="wide")
st.title("Breast Cancer Classification Lab")
st.caption("Compare five classifiers on the same held-out test data.")

uploaded_file = st.file_uploader("Upload test data (CSV)", type=["csv"])
if uploaded_file is None:
    data_path = PROJECT_DIR / "test_data.csv"
    if not data_path.exists():
        st.error("Default test data is missing. Run train_models.py first.")
        st.stop()
    test_data = pd.read_csv(data_path)
    st.caption("Using the project's sample test data. Upload a CSV to evaluate another set.")
else:
    try:
        test_data = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"Could not read the uploaded CSV: {error}")
        st.stop()
    st.caption(f"Evaluating uploaded file: {uploaded_file.name}")

models = {
    name: joblib.load(MODEL_DIR / filename)
    for name, filename in MODEL_FILES.items()
}
feature_columns = list(models["Logistic Regression"].feature_names_in_)
missing_columns = [column for column in feature_columns if column not in test_data.columns]
if missing_columns:
    st.error(f"CSV is missing required feature columns: {', '.join(missing_columns)}")
    st.stop()

features = test_data[feature_columns]
if features.isna().any().any():
    st.error("Feature columns contain missing values. Please clean the CSV and upload again.")
    st.stop()

selected_model_name = st.selectbox("Choose a model", list(MODEL_FILES))
has_labels = TARGET_COLUMN in test_data.columns

if has_labels:
    if test_data[TARGET_COLUMN].isna().any():
        st.error("The diagnosis column contains missing labels.")
        st.stop()

    actual = test_data[TARGET_COLUMN].astype(str)
    expected_labels = {"benign", "malignant"}
    if not set(actual.unique()).issubset(expected_labels):
        st.error("Diagnosis values must be 'benign' or 'malignant'.")
        st.stop()

    comparison_rows = []
    predictions_by_model = {}
    for model_name, model in models.items():
        predictions = model.predict(features)
        positive_index = list(model.classes_).index("malignant")
        positive_scores = model.predict_proba(features)[:, positive_index]
        predictions_by_model[model_name] = predictions
        comparison_rows.append(
            {
                "Model": model_name,
                **calculate_metrics(actual, predictions, positive_scores),
            }
        )

    comparison = pd.DataFrame(comparison_rows).set_index("Model")
    st.subheader("Model comparison")
    st.dataframe(comparison.style.format("{:.3f}"), use_container_width=True)

    selected_predictions = predictions_by_model[selected_model_name]
    selected_metrics = comparison.loc[selected_model_name]
    st.subheader(f"{selected_model_name} on uploaded test data")
    metric_columns = st.columns(6)
    for column, metric_name in zip(
        metric_columns,
        ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
    ):
        column.metric(metric_name, f"{selected_metrics[metric_name]:.3f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        matrix = confusion_matrix(
            actual, selected_predictions, labels=["malignant", "benign"]
        )
        st.dataframe(
            pd.DataFrame(
                matrix,
                index=["Actual malignant", "Actual benign"],
                columns=["Predicted malignant", "Predicted benign"],
            ),
            use_container_width=True,
        )
    with right:
        st.markdown("**Classification report**")
        report = classification_report(
            actual,
            selected_predictions,
            labels=["malignant", "benign"],
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
else:
    selected_model = models[selected_model_name]
    predictions = selected_model.predict(features)
    st.info("No diagnosis column found. Metrics require known test labels.")
    st.dataframe(
        pd.DataFrame({"Prediction": predictions}),
        use_container_width=True,
    )