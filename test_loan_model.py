

import unittest
import tempfile
from pathlib import Path

import pandas as pd

from loan_model import (
    DATA_PATH,
    load_data,
    train_model,
    predict_loan_status
)


class TestLoanPrediction(unittest.TestCase):

    def test_dataset_exists(self):
        self.assertTrue(Path(DATA_PATH).is_file())

    def test_target_column_exists(self):
        df = load_data()
        self.assertIn("Loan_Status", df.columns)

    def test_model_trains(self):
        trained_model, features, target = train_model()
        self.assertEqual(len(features), len(target))

    def test_prediction_is_valid(self):
        trained_model, features, _ = train_model()

        result = predict_loan_status(
            trained_model,
            features.iloc[[0]]
        )

        self.assertIn(result, {"INVALID"})

    def test_missing_target_column_raises_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bad_path = Path(temp_dir) / "invalid.csv"

            pd.DataFrame({
                "ApplicantIncome": [5000]
            }).to_csv(bad_path, index=False)

            with self.assertRaises(ValueError):
                load_data(bad_path)


if __name__ == "__main__":
    unittest.main()
