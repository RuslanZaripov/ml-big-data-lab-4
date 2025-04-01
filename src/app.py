from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from logger import Logger
from database import RedisClient
from kafka_producer import KafkaProducer
import traceback
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
    
    def __init__(self):
        """
        Initializes the web application, loads the model, and creates the FastAPI app.
        
        Args:
            args (argparse.Namespace): Parsed command-line arguments.
        """
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)

        self.app = self._create_app()
        self.log.info('FastAPI app initialized')

        self.database_client = RedisClient()
        
        self.kafka_producer = KafkaProducer(
            broker="broker:9092",
            topic="predictions",
        )

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
                prediction_id = str(uuid.uuid4())
                
                prediction_data = {
                    "prediction_id": prediction_id,
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

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Runs the FastAPI application using Uvicorn.
        
        Args:
            host (str): Host address to run the server.
            port (int): Port number to run the server.
        """
        uvicorn.run(self.app, host=host, port=port)

if __name__ == "__main__":
    web_app = WebApp()
    web_app.run()
