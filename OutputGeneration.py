import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from typing import Any

load_dotenv()

def GenerateOutput(Prompt: str, **Variables: Any) -> str:
    """
    Generate output using Azure OpenAI with f-string formatted prompt.
    
    Args:
        Prompt: The prompt template with f-string placeholders
        **Variables: All variables required for f-string formatting
    
    Returns:
        The generated output from Azure OpenAI
    """
    # Format the prompt with provided variables
    FormattedPrompt = Prompt.format(**Variables)
    
    # Initialize Azure OpenAI client
    Client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    # Generate response
    Response = Client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=[
            {"role": "user", "content": FormattedPrompt}
        ]
    )
    
    return Response.choices[0].message.content