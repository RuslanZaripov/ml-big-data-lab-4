import json
from logger import Logger
from confluent_kafka import Producer

SHOW_LOG = True


class KafkaProducer:
    def __init__(
        self, 
        bootstrap_servers: str, 
        topic: str
    ):
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)
        producer_config = {
            'bootstrap.servers': bootstrap_servers,
        }
        self.producer = Producer(producer_config)
        self.topic = topic
        
    def delivery_report(self, err, msg):
        if err is not None:
            self.log.error(f'Message delivery failed: {err}')
        else:
            self.log.info(f'Message delivered to {msg.topic()} [{msg.partition()}]')

    def send_message(self, message: dict):
        try:
            self.producer.poll(0)
            self.producer.produce(
                self.topic,
                json.dumps(message).encode('utf-8'),
                callback=self.delivery_report
            )
            self.producer.flush()
        except Exception as e:
            self.log.error(f'Failed to send message: {e}')
            raise
