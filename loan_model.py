
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


# Dataset path relative to this Python file
DATA_PATH = (
    Path(__file__).resolve().parent
    / "train_u6luX_CVtuZ9i (1).csv"
)

# Output files are saved in the project folder
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "loan_model.pkl"
METRICS_PATH = PROJECT_DIR / "metrics.json"


def load_data(data_path=DATA_PATH):
    """Load and validate the Loan Prediction dataset."""
    df = pd.read_csv(data_path)

    if "Loan_Status" not in df.columns:
        raise ValueError("Dataset must contain Loan_Status")

    df = df.dropna(subset=["Loan_Status"]).copy()

    if df.empty:
        raise ValueError("No labelled training records found")

    return df


def build_model(X):
    """Build preprocessing and Logistic Regression pipeline."""
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

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    return model


def train_model(data_path=DATA_PATH):
    """Train the model using all labelled records."""
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
    """Predict loan approval status: Y or N."""
    prediction = model.predict(applicant)[0]
    return str(prediction)


def train_evaluate_and_save(data_path=DATA_PATH):
    """Train, evaluate, and save the model and its metrics."""
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

    print("Training model on the training split...")

    model = build_model(X_train)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=["N", "Y"]
    )

    # Save the trained model
    joblib.dump(model, MODEL_PATH)

    # Save evaluation metrics for Practical 3
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

    print("\nModel saved as:", MODEL_PATH.name)
    print("Metrics saved as:", METRICS_PATH.name)

    return accuracy


if __name__ == "__main__":
    # Train and evaluate on a separate test set.
    train_evaluate_and_save()

    # Demonstrate prediction using the saved model.
    saved_model = joblib.load(MODEL_PATH)
    df = load_data()

    sample = df.drop(
        columns=["Loan_Status", "Loan_ID"],
        errors="ignore"
    ).iloc[[0]]

    prediction = predict_loan_status(saved_model, sample)

    print("\nSample prediction:", prediction)
