import unittest
import json
import sys
import os
import argparse
from fastapi.testclient import TestClient

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from app import WebApp

class TestPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = argparse.Namespace(model="LOG_REG")
        cls.app = WebApp(args).app
        cls.client = TestClient(cls.app)

    def test_prediction_and_retrieval(self):
        with open("tests/test_0.json", "r") as file:
            input_data = json.load(file)

        response = self.client.post("/predict", json=input_data)
        self.assertEqual(response.status_code, 200)

        prediction_id = response.json()["prediction_id"]
        self.assertIsNotNone(prediction_id)
        
        response = self.client.get(f"/predictions/{prediction_id}")
        self.assertEqual(response.status_code, 200)

        retrieved_data = response.json()
        
        self.assertIn("prediction", retrieved_data)
        self.assertIn("score", retrieved_data)
        self.assertEqual(json.loads(retrieved_data["input_data"]), input_data)

if __name__ == "__main__":
    unittest.main()