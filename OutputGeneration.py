import os
import asyncio
from openai import AzureOpenAI, AsyncAzureOpenAI
from dotenv import load_dotenv
from typing import Any, List, Dict

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
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[
            {"role": "user", "content": FormattedPrompt}
        ]
    )
    
    return Response.choices[0].message.content


async def GenerateOutputAsync(Prompt: str, **Variables: Any) -> str:
    """
    Generate output using Azure OpenAI with f-string formatted prompt asynchronously.
    
    Args:
        Prompt: The prompt template with f-string placeholders
        **Variables: All variables required for f-string formatting
    
    Returns:
        The generated output from Azure OpenAI
    """
    # Format the prompt with provided variables
    FormattedPrompt = Prompt.format(**Variables)
    
    # Initialize Azure OpenAI async client
    Client = AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    # Generate response
    Response = await Client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[
            {"role": "user", "content": FormattedPrompt}
        ]
    )
    
    await Client.close()
    return Response.choices[0].message.content


async def GenerateOutputBatch(PromptVariablePairs: List[Dict[str, Any]]) -> List[str]:
    """
    Generate multiple outputs concurrently using Azure OpenAI.
    
    Args:
        PromptVariablePairs: List of dictionaries containing 'Prompt' and variables
        
    Returns:
        List of generated outputs from Azure OpenAI
    """
    Tasks = []
    for Item in PromptVariablePairs:
        Prompt = Item['Prompt']
        Variables = {k: v for k, v in Item.items() if k != 'Prompt'}
        Tasks.append(GenerateOutputAsync(Prompt, **Variables))
    
    Results = await asyncio.gather(*Tasks)
    return Results