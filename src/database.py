import os
from redis import Redis, exceptions
from dotenv import load_dotenv
from logger import Logger

SHOW_LOG = True

class RedisClient:
    """
    Redis client wrapper for connecting, setting, getting, and managing key-value pairs in Redis.
    """
    def __init__(self):
        """
        Initializes the Redis client, loads environment variables, and attempts connection.
        """
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)

        load_dotenv(dotenv_path='env/.env', override=True)

        self.client = Redis(
            host=os.getenv('REDIS_HOST'),
            port=int(os.getenv('REDIS_PORT')),
            db=int(os.getenv('REDIS_DB')),
            password=os.getenv('REDIS_PASSWORD'),
            decode_responses=True
        )
        self.try_connect()

    def try_connect(self):
        """
        Attempts to connect to the Redis server and logs the connection status.
        
        Returns:
            bool: True if connection is successful, False otherwise.
        """
        try:
            info = self.client.info()
            self.log.info(f"redis_version={info['redis_version']}")
            response = self.client.ping()
            if response:
                self.log.info("Connection successful!")
            else:
                self.log.warning("Failed to connect to Redis.")
                return False
        except exceptions.RedisError as e:
            self.log.error(f"Error: {e}")
            return False
        
        return True

    def set(self, key, value, ex=None):
        """
        Sets a key-value pair in Redis with an optional expiration time.
        
        Args:
            key (str): The key to store the value under.
            value (str): The value to store.
            ex (int, optional): Expiration time in seconds.
        
        Returns:
            bool: True if the operation is successful.
        """
        return self.client.set(key, value, ex=ex)
    
    def get(self, key):
        """
        Retrieves a value from Redis by key.
        
        Args:
            key (str): The key to retrieve.
        
        Returns:
            str or None: The value if the key exists, otherwise None.
        """
        return self.client.get(key)

    def delete(self, key):
        """
        Deletes a key from Redis.
        
        Args:
            key (str): The key to delete.
        
        Returns:
            bool: True if the key was deleted, False otherwise.
        """
        return self.client.delete(key) == 1
    
    def exists(self, key):
        """
        Checks if a key exists in Redis.
        
        Args:
            key (str): The key to check.
        
        Returns:
            bool: True if the key exists, False otherwise.
        """
        return self.client.exists(key) == 1

    def close(self):
        """
        Closes the Redis client connection.
        """
        if self.client:
            self.client.close()

if __name__ == "__main__":
    redis_client = RedisClient()
    redis_client.close()
