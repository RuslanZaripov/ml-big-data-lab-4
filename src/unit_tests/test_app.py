import unittest
import json
import sys
import os
import time
from fastapi.testclient import TestClient

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from logger import Logger
from app import WebApp


class TestPrediction(unittest.TestCase):
    SHOW_LOG = True
     
    @classmethod
    def setUpClass(cls):
        logger = Logger(cls.SHOW_LOG)
        cls.log = logger.get_logger(__name__)

        cls.app = WebApp().app
        cls.client = TestClient(cls.app)

    def test_prediction_and_retrieval(self):
        with open("tests/test_0.json", "r") as file:
            input_data = json.load(file)

        response = self.client.post("/predict", json=input_data)
        self.assertEqual(response.status_code, 200)
        self.log.info(f"Prediction POST response: {response.json()}")

        prediction_id = response.json()["prediction_id"]
        self.assertIsNotNone(prediction_id)
        self.log.info(f"Received prediction_id: {prediction_id}")
        
        max_retries = 5
        for attempt in range(max_retries):
            self.log.info(f"Attempt {attempt + 1}/{max_retries} to retrieve prediction")
            response = self.client.get(f"/predictions/{prediction_id}")
            if response.status_code == 200:
                self.log.info("Successfully retrieved prediction")
                break
            time.sleep(0.5)
        
        self.assertEqual(response.status_code, 200)

        retrieved_data = response.json()
        self.log.info(f"Retrieved data: {retrieved_data}")
        self.assertIn("prediction", retrieved_data)
        self.assertIn("score", retrieved_data)
        self.assertEqual(json.loads(retrieved_data["input_data"]), input_data)

if __name__ == "__main__":
    unittest.main()