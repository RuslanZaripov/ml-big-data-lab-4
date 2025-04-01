from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from logger import Logger
from database import RedisClient
from kafka_consumer import KafkaConsumer
from kafka_producer import KafkaProducer
import time
import threading
import pandas as pd
import traceback
import sys
import pickle
import configparser
import argparse
import uvicorn
import json
import uuid

SHOW_LOG = True


class PredictionInput(BaseModel):
    """
    Pydantic model for input data validation.
    
    Attributes:
        X (List[Dict[str, float]]): List of feature dictionaries.
        y (List[Dict[str, float]]): List of target dictionaries.
    """
    X: List[Dict[str, float]]
    y: List[Dict[str, float]]

class WebApp:
    """
    Web application class using FastAPI for serving a machine learning model.
    """
    
    def __init__(self, args):
        """
        Initializes the web application, loads the model, and creates the FastAPI app.
        
        Args:
            args (argparse.Namespace): Parsed command-line arguments.
        """
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)

        self.config = configparser.ConfigParser()
        self.config.read("config.ini")

        self.args = args
        self.model, self.scaler = self._load_model()
        self.log.info('Web app model initialized')

        self.app = self._create_app()
        self.log.info('FastAPI app initialized')

        self.database_client = RedisClient()
        
        # Initialize Kafka
        self.kafka_producer = KafkaProducer(
            bootstrap_servers="kafka:29092",
            topic="predictions",
        )
        
        # Start Kafka consumer in a separate thread
        self.kafka_consumer = KafkaConsumer(
            bootstrap_servers="kafka:29092",
            topic="predictions",
            group_id="ml_app_group"
        )
        
        self.consumer_thread = threading.Thread(
            target=self._consume_predictions,
            daemon=True
        )
        self.consumer_thread.start()
        
        self._shutdown_event = threading.Event()
        
    def _consume_predictions(self):
        """Background thread to consume prediction messages"""
        while not self._shutdown_event.is_set():
            message = self.kafka_consumer.consume_messages(timeout=10.0)

            if not message:
                continue
            
            self.log.info(f"Received prediction result: {message}")
            
            try:
                prediction_id = message['prediction_id']
                
                if self.database_client.get(prediction_id):
                    self.log.warning(f"Prediction {prediction_id} already exists in database")
                else:
                    self.database_client.set(
                        prediction_id,
                        json.dumps(message)
                    )
                    self.log.info(f"Stored prediction {prediction_id} in database")
                    
            except Exception as db_error:
                self.log.error(f"Database operation failed: {str(db_error)}")
                self.log.error(traceback.format_exc())

    def _create_app(self):
        """
        Creates and configures the FastAPI application.
        
        Returns:
            FastAPI: Configured FastAPI app instance.
        """
        app = FastAPI()

        @app.get("/")
        async def root():
            """Root endpoint for health check."""
            return {"message": "Hello World"}

        @app.post("/predict")
        async def predict(input_data: PredictionInput):
            """
            Endpoint for making predictions using the trained model.
            
            Args:
                input_data (PredictionInput): Input data containing X (features) and y (target values).
            
            Returns:
                dict: A dictionary containing predictions and model score.
            """
            try:
                X = self.scaler.transform(pd.json_normalize(input_data.X))
                y = pd.json_normalize(input_data.y)
                score = self.model.score(X, y)
                pred = self.model.predict(X).tolist()

                prediction_id = str(uuid.uuid4())
                prediction_data = {
                    "prediction": pred,
                    "score": score,
                    "input_data": input_data.model_dump_json()
                }
                
                self.kafka_producer.send_message(prediction_data)
                
                return {"prediction_id": prediction_id}
            except Exception as e:
                self.log.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=str(e))
            
        @app.get("/predictions/{prediction_id}")
        async def get_prediction(prediction_id: str):
            """
            Endpoint for retrieving a prediction from Redis by its ID.
            
            Args:
                prediction_id (str): The ID of the prediction to retrieve.
            
            Returns:
                dict: The prediction data stored in Redis.
            """
            prediction_data = self.database_client.get(prediction_id)
            if prediction_data is None:
                raise HTTPException(status_code=404, detail="Prediction not found")
            return json.loads(prediction_data)

        return app

    def _load_model(self):
        """
        Loads the machine learning model and scaler from disk.
        
        Returns:
            tuple: Loaded model and scaler.
        """
        try:
            with open(self.config[self.args.model]["path"], "rb") as model_file:
                model = pickle.load(model_file)
            
            with open(self.config["STD_SCALER"]["path"], "rb") as scaler_file:
                scaler = pickle.load(scaler_file)
        
            return model, scaler
        except FileNotFoundError:
            self.log.error(traceback.format_exc())
            sys.exit(1)

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Runs the FastAPI application using Uvicorn.
        
        Args:
            host (str): Host address to run the server.
            port (int): Port number to run the server.
        """
        try:
            uvicorn.run(self.app, host=host, port=port)
        finally:
            self._shutdown_event.set()
            self.kafka_consumer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web App Model")
    parser.add_argument("-m", "--model",
                        type=str,
                        help="Select model",
                        required=True,
                        default="LOG_REG",
                        const="LOG_REG",
                        nargs="?",
                        choices=["LOG_REG"])
    args = parser.parse_args()

    web_app = WebApp(args)
    web_app.run()
