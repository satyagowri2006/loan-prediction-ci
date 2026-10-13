
import json
import sys
from pathlib import Path

METRICS_PATH = Path(__file__).resolve().parent / "metrics.json"
MINIMUM_ACCURACY = 0.85

print("Reading model evaluation metrics...")

if not METRICS_PATH.is_file():
    print("QUALITY GATE FAILED")
    print("metrics.json was not found. Run loan_model.py first.")
    sys.exit(1)

with open(METRICS_PATH, "r", encoding="utf-8") as file:
    metrics = json.load(file)

accuracy = metrics["accuracy"]

print(f"Model Accuracy: {accuracy:.4f}")
print(f"Required Accuracy: {MINIMUM_ACCURACY:.2f}")

if accuracy < MINIMUM_ACCURACY:
    print("QUALITY GATE FAILED")
    print("Model performance is below the required threshold.")
    sys.exit(1)

print("QUALITY GATE PASSED")
print("Model performance satisfies the required threshold.")
sys.exit(0)
