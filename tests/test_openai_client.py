import importlib
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import OpenAiClient

class OpenAiClientSingletonTest(unittest.TestCase):
    def test_get_client_singleton(self):
        OpenAiClient.GetClient.cache_clear()
        DummyResponse = MagicMock()
        DummyResponse.choices = [MagicMock(message=MagicMock(content="ok"))]
        DummyClient = MagicMock()
        DummyClient.chat.completions.create.return_value = DummyResponse
        with patch('OpenAiClient.AzureOpenAI', return_value=DummyClient) as MockAzure:
            Client1 = OpenAiClient.GetClient()
            Client2 = OpenAiClient.GetClient()
        self.assertIs(Client1, Client2)
        MockAzure.assert_called_once()

    def test_generate_output_uses_singleton(self):
        import OutputGeneration
        importlib.reload(OutputGeneration)
        OpenAiClient.GetClient.cache_clear()
        DummyResponse = MagicMock()
        DummyResponse.choices = [MagicMock(message=MagicMock(content="ok"))]
        DummyClient = MagicMock()
        DummyClient.chat.completions.create.return_value = DummyResponse
        with patch('OpenAiClient.AzureOpenAI', return_value=DummyClient) as MockAzure:
            OutputGeneration.GenerateOutput('hello')
            OutputGeneration.GenerateOutput('world')
            self.assertEqual(MockAzure.call_count, 1)

if __name__ == '__main__':
    unittest.main()
