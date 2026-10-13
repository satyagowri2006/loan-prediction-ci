
import unittest

from app import app


class TestLoanPredictionAPI(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["status"],
            "ok"
        )

    def test_prediction_endpoint(self):
        response = self.client.post(
            "/predict",
            json={
                "Gender": "Male",
                "Married": "Yes",
                "Dependents": "0",
                "Education": "Graduate",
                "Self_Employed": "No",
                "ApplicantIncome": 5000,
                "CoapplicantIncome": 0,
                "LoanAmount": 150,
                "Loan_Amount_Term": 360,
                "Credit_History": 1,
                "Property_Area": "Urban"
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            response.get_json()["prediction"],
            {"Y", "N"}
        )

    def test_missing_fields_return_400(self):
        response = self.client.post(
            "/predict",
            json={"Gender": "Male"}
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "missing_fields",
            response.get_json()
        )


if __name__ == "__main__":
    unittest.main()
