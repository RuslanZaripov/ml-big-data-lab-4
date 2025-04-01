import json
from logger import Logger
from confluent_kafka import Consumer

SHOW_LOG = True


class KafkaConsumer:
    def __init__(
        self, 
        bootstrap_servers: str, 
        topic: str, 
        group_id: str
    ):
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)
        consumer_config = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'earliest'
        }
        self.consumer = Consumer(consumer_config)
        self.consumer.subscribe([topic])

    def consume_messages(self, timeout: float = 1.0):
        try:
            msg = self.consumer.poll(timeout)
            
            if msg is None:
                return None
            
            if msg.error():
                self.log.error(f"Consumer error: {msg.error()}")
                return None
            
            return json.loads(msg.value().decode('utf-8'))
        
        except Exception as e:
            self.log.error(f"Error consuming message: {e}")
            return None

    def close(self):
        self.consumer.close()