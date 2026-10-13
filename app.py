
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "loan_model.pkl"

FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]


def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "loan_model.pkl not found. Run loan_model.py first."
        )
    return joblib.load(MODEL_PATH)


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "loan-prediction"
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "A JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    try:
        applicant = pd.DataFrame(
            [{feature: data[feature] for feature in FEATURES}]
        )
        model = load_model()
        prediction = str(model.predict(applicant)[0])

    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500

    except (TypeError, ValueError) as exc:
        return jsonify({
            "error": f"Invalid input: {exc}"
        }), 400

    return jsonify({
        "prediction": prediction,
        "meaning": (
            "Loan Approved" if prediction == "Y"
            else "Loan Not Approved"
        )
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
