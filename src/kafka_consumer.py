import json
import pandas as pd
import pickle
import traceback
import sys
import argparse
import configparser
from logger import Logger
from database import RedisClient
from confluent_kafka import Consumer, KafkaError

SHOW_LOG = True


class KafkaConsumer:
    def __init__(
        self, 
        args: argparse.Namespace,
        broker: str, 
        topic: str, 
        group: str
    ):
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)
        
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")
        
        self.args = args
        
        self.model, self.scaler = self._load_model()
        self.log.info('Kafka Consumer Model initialized')
        
        self.database_client = RedisClient()
        
        conf = {
            'bootstrap.servers': broker, 
            'group.id': group, 
            'session.timeout.ms': 6000,
            'auto.offset.reset': 'earliest', 
            'enable.auto.offset.store': False}
        
        self.consumer = Consumer(conf)
        self.consumer.subscribe([topic])
        
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
    
    def process(self, data: dict):
        prediction_id = data['prediction_id']
        input_data = data['input_data'] 
            
        X = self.scaler.transform(pd.json_normalize(input_data['X']))
        pred = self.model.predict(X).tolist()
        
        prediction_data = {'pred': pred}
        
        self.database_client.set(prediction_id, prediction_data)
        
    def consume_messages(self, timeout: float = 1.0):
        try:
            while True:
                msg = self.consumer.poll(timeout)

                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        self.log.error(f'Reached end of partition: {msg.topic()}[{msg.partition()}]')
                    else:
                        self.log.error(f'Error while consuming messages: {msg.error()}')
                    continue
                
                self.log.info(f"Received message on consumer one: {msg.value().decode('utf-8')}")
                
                self.process(json.loads(msg.value().decode('utf-8')))

        except KeyboardInterrupt:
            self.log.info("Stopping consumer...")

        finally:
            self.consumer.close()
            self.log.info("Consumer closed")
            
        
def main(args):    
    consumer = KafkaConsumer(
        args,
        broker="broker:9092",
        topic="predictions",
        group="ml_app_group"
    )
    consumer.consume_messages()

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kafka Consumer")
    parser.add_argument("-m", "--model",
                        type=str,
                        help="Select model",
                        required=True,
                        default="LOG_REG",
                        const="LOG_REG",
                        nargs="?",
                        choices=["LOG_REG"])
    args = parser.parse_args()
    main(args)
