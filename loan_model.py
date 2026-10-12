
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Dataset path relative to the Python file
DATA_PATH = (
    Path(__file__).resolve().parent
    / "train_u6lujuX_CVtuZ9i (1).csv"
)


def load_data(data_path=DATA_PATH):
    df = pd.read_csv(data_path)

    if "Loan_Status" not in df.columns:
        raise ValueError("Dataset must contain Loan_Status")

    df = df.dropna(subset=["Loan_Status"]).copy()

    if df.empty:
        raise ValueError("No labelled training records found")

    return df


def train_model(data_path=DATA_PATH):
    df = load_data(data_path)

    X = df.drop(
        columns=["Loan_Status", "Loan_ID"],
        errors="ignore"
    )
    y = df["Loan_Status"].astype(str).str.strip()

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

    model.fit(X, y)

    return model, X, y


def predict_loan_status(model, applicant):
    prediction = model.predict(applicant)[0]
    return str(prediction)


if __name__ == "__main__":
    model, X, y = train_model()

    prediction = predict_loan_status(
        model, X.iloc[[0]]
    )

    print("Model trained successfully!")
    print("Training rows:", len(X))
    print("Sample prediction:", prediction)
