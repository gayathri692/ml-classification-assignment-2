from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
]
RAW_COLUMNS = ["id", "diagnosis", *FEATURE_COLUMNS]


def main():
    raw_data = pd.read_csv(
        Path(__file__).with_name("wdbc.data"),
        header=None,
        names=RAW_COLUMNS,
    )
    data = raw_data.drop(columns=["id"])
    data["diagnosis"] = data["diagnosis"].map({"M": "malignant", "B": "benign"})
    if data["diagnosis"].isna().any():
        raise ValueError("The raw data contains an unknown diagnosis label.")

    feature_columns = [column for column in data.columns if column != "diagnosis"]
    class_counts = data["diagnosis"].value_counts()

    print(f"Rows: {len(data)}")
    print(f"Features: {len(feature_columns)}")
    print("Class counts:")
    print(class_counts.to_string())
    print(f"Missing values: {int(data.isna().sum().sum())}")

    if len(data) < 500 or len(feature_columns) < 12:
        raise ValueError("The selected dataset does not meet the assignment minimums.")

    output_path = Path(__file__).with_name("breast_cancer_data.csv")
    data.to_csv(output_path, index=False)
    print(f"Dataset saved to: {output_path}")


if __name__ == "__main__":
    main()