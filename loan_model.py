
from pathlib import Path
import json
import joblib

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "loan_prediction.csv"
MODEL_PATH = PROJECT_DIR / "loan_model.pkl"
METRICS_PATH = PROJECT_DIR / "metrics.json"


def load_data(data_path=DATA_PATH):
    df = pd.read_csv(data_path)

    if "Loan_Status" not in df.columns:
        raise ValueError("Dataset must contain Loan_Status")

    df = df.dropna(subset=["Loan_Status"]).copy()

    if df.empty:
        raise ValueError("No labelled training records found")

    return df


def build_model(X):
    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(max_iter=1000))
    ])


def train_model(data_path=DATA_PATH):
    df = load_data(data_path)

    X = df.drop(
        columns=["Loan_Status", "Loan_ID"],
        errors="ignore"
    )
    y = df["Loan_Status"].astype(str).str.strip()

    model = build_model(X)
    model.fit(X, y)

    return model, X, y


def predict_loan_status(model, applicant):
    return str(model.predict(applicant)[0])


def train_evaluate_and_save(data_path=DATA_PATH):
    df = load_data(data_path)

    X = df.drop(
        columns=["Loan_Status", "Loan_ID"],
        errors="ignore"
    )
    y = df["Loan_Status"].astype(str).str.strip()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = build_model(X_train)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(
        y_test, predictions, labels=["N", "Y"]
    )

    joblib.dump(model, MODEL_PATH)

    metrics = {
        "accuracy": float(accuracy),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "confusion_matrix": matrix.tolist()
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nModel Evaluation")
    print("----------------")
    print(f"Test accuracy: {accuracy:.2%}")
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Confusion matrix (labels N, Y):")
    print(matrix)
    print("Model saved as:", MODEL_PATH.name)
    print("Metrics saved as:", METRICS_PATH.name)

    return accuracy


if __name__ == "__main__":
    train_evaluate_and_save()
