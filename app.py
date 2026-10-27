
import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "loan_model.pkl"
LOG_PATH = PROJECT_DIR / "loan_prediction.log"

# Configure logging to both a file and the console.
logger = logging.getLogger("loan_prediction")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    file_handler = RotatingFileHandler(
        LOG_PATH,
        maxBytes=1_000_000,
        backupCount=3,
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

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


@app.before_request
def log_request():
    logger.info(
        "Request received: method=%s path=%s",
        request.method,
        request.path,
    )


@app.get("/")
def health_check():
    logger.info("Health check successful")
    return jsonify({
        "status": "ok",
        "service": "loan-prediction",
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        logger.warning("Prediction rejected: missing or invalid JSON")
        return jsonify({
            "error": "A JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        logger.warning(
            "Prediction rejected: missing fields=%s",
            missing_fields,
        )
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields,
        }), 400

    try:
        applicant = pd.DataFrame(
            [{feature: data[feature] for feature in FEATURES}]
        )
        model = load_model()
        prediction = str(model.predict(applicant)[0])

        logger.info("Prediction completed successfully")

        return jsonify({
            "prediction": prediction,
            "meaning": (
                "Loan Approved" if prediction == "Y"
                else "Loan Not Approved"
            ),
        })

    except FileNotFoundError:
        logger.exception("Prediction failed: model file unavailable")
        return jsonify({
            "error": "Prediction model is unavailable"
        }), 500

    except (TypeError, ValueError) as exc:
        logger.warning("Prediction rejected: invalid input: %s", exc)
        return jsonify({
            "error": f"Invalid input: {exc}"
        }), 400

    except Exception:
        logger.exception("Unexpected error during prediction")
        return jsonify({
            "error": "An internal error occurred"
        }), 500


@app.errorhandler(404)
def not_found(error):
    logger.warning("Route not found: path=%s", request.path)
    return jsonify({"error": "Route not found"}), 404


@app.errorhandler(500)
def internal_server_error(error):
    logger.exception("Internal server error: %s", error)
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
