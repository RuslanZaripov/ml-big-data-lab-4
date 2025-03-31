import unittest
import sys
import os

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from database import RedisClient


class TestRedisClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.redis_client = RedisClient()

    @classmethod
    def tearDownClass(cls):
        cls.redis_client.close()

    def setUp(self):
        self.test_key = "test_key"
        self.test_value = "test_value"

    def tearDown(self):
        if self.redis_client.exists(self.test_key):
            self.redis_client.delete(self.test_key)

    def test_connection(self):
        self.assertTrue(self.redis_client.try_connect())

    def test_set_and_get(self):
        self.assertTrue(self.redis_client.set(self.test_key, self.test_value))
        
        retrieved_value = self.redis_client.get(self.test_key)
        self.assertEqual(retrieved_value, self.test_value)

    def test_delete(self):
        self.redis_client.set(self.test_key, self.test_value)
        
        self.assertTrue(self.redis_client.delete(self.test_key))
        self.assertFalse(self.redis_client.exists(self.test_key))

    def test_exists(self):
        self.redis_client.set(self.test_key, self.test_value)
        self.assertTrue(self.redis_client.exists(self.test_key))
        
        self.redis_client.delete(self.test_key)
        self.assertFalse(self.redis_client.exists(self.test_key))

if __name__ == "__main__":
    unittest.main()
