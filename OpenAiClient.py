import os
from functools import lru_cache
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

@lru_cache(maxsize=1)
def GetClient() -> AzureOpenAI:
    """Return a shared AzureOpenAI client instance."""
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
