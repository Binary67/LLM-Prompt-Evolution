import pandas as pd
import os
import asyncio
import re
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
import time

load_dotenv()

async def RetryAzureOpenAICall(CallFunction, MaxRetries=25, InitialDelay=1):
    """
    Retry Azure OpenAI API calls with exponential backoff.
    
    Args:
        CallFunction: Async function to call
        MaxRetries: Maximum number of retry attempts (default: 25)
        InitialDelay: Initial delay in seconds (default: 1)
    
    Returns:
        The result of the successful API call
    
    Raises:
        Exception: If all retry attempts fail
    """
    Delay = InitialDelay
    LastException = None
    
    for Attempt in range(MaxRetries + 1):
        try:
            Result = await CallFunction()
            return Result
        except Exception as e:
            LastException = e
            if Attempt < MaxRetries:
                print(f"Azure OpenAI call failed (attempt {Attempt + 1}/{MaxRetries + 1}): {e}")
                print(f"Retrying in {Delay} seconds...")
                await asyncio.sleep(Delay)
                # Exponential backoff with cap at 60 seconds
                Delay = min(Delay * 2, 60)
            else:
                print(f"Azure OpenAI call failed after {MaxRetries + 1} attempts")
                raise LastException

def ExtractLabelFromPrediction(Prediction, TargetLabel):
    """
    Extract label from prediction using simple exact word matching.
    
    Args:
        Prediction (str): The raw model prediction text
        TargetLabel (list): List of target labels to match against
    
    Returns:
        str: The extracted label or None if no match found
    """
    if not Prediction:
        return None
    
    PredictionLower = Prediction.lower().strip()
    
    # Check for each target label in the prediction
    for Label in TargetLabel:
        LabelLower = Label.lower()
        
        # Use regex to find exact word match
        if re.search(rf'\b{re.escape(LabelLower)}\b', PredictionLower):
            return Label
    
    return None

async def EvaluatePrompt(Prompt, Dataframe, TargetLabel):
    Client = AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    async def ProcessRow(Index, Row):
        FormattedPrompt = Prompt.format(text=Row['text'])
        
        try:
            async def MakeAPICall():
                return await Client.chat.completions.create(
                    model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                    messages=[
                        {"role": "user", "content": FormattedPrompt}
                    ],
                )
            
            Response = await RetryAzureOpenAICall(MakeAPICall)
            Prediction = Response.choices[0].message.content.strip()
            return Index, Prediction
            
        except Exception as e:
            print(f"Error processing row {Index}: {e}")
            return Index, "Error"
    
    Tasks = [ProcessRow(Index, Row) for Index, Row in Dataframe.iterrows()]
    Results = await asyncio.gather(*Tasks)
    
    Predictions = {}
    for Index, Prediction in Results:
        Predictions[Index] = Prediction
    
    PredictionsList = [Predictions[Index] for Index in Dataframe.index]
    
    Dataframe['ModelPrediction'] = PredictionsList
    
    # Extract labels from predictions using regex
    ExtractedLabels = []
    for Prediction in PredictionsList:
        ExtractedLabel = ExtractLabelFromPrediction(Prediction, TargetLabel)
        ExtractedLabels.append(ExtractedLabel)
    
    Dataframe['ExtractedLabel'] = ExtractedLabels
    print(Dataframe['ExtractedLabel'].unique())
    
    # Calculate accuracy using extracted labels
    CorrectPredictions = sum(1 for ExtractedLabel, Label in zip(ExtractedLabels, Dataframe['label']) 
                           if ExtractedLabel is not None and ExtractedLabel.lower() == Label.lower())
    Accuracy = CorrectPredictions / len(Dataframe)
    
    return Accuracy, Dataframe