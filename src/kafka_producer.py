import json
from logger import Logger
from confluent_kafka import Producer

SHOW_LOG = True


class KafkaProducer:
    def __init__(
        self, 
        broker: str, 
        topic: str
    ):
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)
        
        producer_config = {
            'bootstrap.servers': broker,
        }
        self.topic = topic
        
        self.producer = Producer(producer_config)
        
    def delivery_report(self, err, msg):
        if err is not None:
            self.log.error(f'Message delivery failed: {err}')
        else:
            self.log.info(f'Message delivered to {msg.topic()} [{msg.partition()}]')

    def send_message(self, message: dict):
        try:
            data = json.dumps(message).encode('utf-8') 
            self.producer.produce(
                self.topic,
                value=data,
                callback=self.delivery_report
            )
            self.producer.flush()
            
        except Exception as e:
            self.log.error(f'Failed to send message: {e}')
            raise
