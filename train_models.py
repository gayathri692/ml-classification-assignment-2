from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


PROJECT_DIR = Path(__file__).parent
TARGET_COLUMN = "diagnosis"
POSITIVE_LABEL = "malignant"
RANDOM_STATE = 42


def build_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "K-Nearest Neighbors": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=5)
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": make_pipeline(
            StandardScaler(), SVC(probability=True, random_state=RANDOM_STATE)
        ),
    }


def calculate_metrics(y_true, y_pred, positive_scores):
    actual_positive = np.asarray(y_true) == POSITIVE_LABEL

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(actual_positive, positive_scores),
        "Precision": precision_score(
            y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0
        ),
        "F1": f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    data = pd.read_csv(PROJECT_DIR / "breast_cancer_data.csv")
    features = data.drop(columns=[TARGET_COLUMN])
    target = data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    test_data = x_test.copy()
    test_data[TARGET_COLUMN] = y_test
    test_data.to_csv(PROJECT_DIR / "test_data.csv", index=False)

    model_dir = PROJECT_DIR / "model"
    model_dir.mkdir(exist_ok=True)
    results = []

    for model_name, model in build_models().items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        positive_index = list(model.classes_).index(POSITIVE_LABEL)
        positive_scores = model.predict_proba(x_test)[:, positive_index]

        results.append({"Model": model_name, **calculate_metrics(
            y_test, predictions, positive_scores
        )})
        model_file = model_name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, model_dir / model_file)

    results_frame = pd.DataFrame(results)
    results_frame.to_csv(model_dir / "metrics.csv", index=False)
    print(results_frame.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nTest data saved to: {PROJECT_DIR / 'test_data.csv'}")
    print(f"Saved models and metrics to: {model_dir}")


if __name__ == "__main__":
    main()