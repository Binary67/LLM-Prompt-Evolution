import pandas as pd
import os
import asyncio
import re
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

load_dotenv()

def ExtractLabelFromPrediction(Prediction, ValidLabels):
    """
    Extract label from prediction using a simple pattern matching.
    
    Args:
        Prediction (str): The raw model prediction text
        ValidLabels (list): List of valid labels to match against
    
    Returns:
        str: The extracted label or None if no match found
    """
    if not Prediction:
        return None
    
    PredictionLower = Prediction.lower().strip()
    
    # Check for specific patterns with flexibility for underscores/spaces
    if re.search(r'\bhas[_\s]*aspiration\b', PredictionLower):
        return 'has_aspiration'
    elif re.search(r'\bno[_\s]*aspiration\b', PredictionLower):
        return 'no_aspiration'
    
    return None

async def EvaluatePrompt(Prompt, Dataframe):
    Client = AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    if "{text}" not in Prompt:
        Prompt = Prompt + " Here is the talent feedback: {text}"
    
    async def ProcessRow(Index, Row):
        FormattedPrompt = Prompt.format(text=Row['text'])
        
        try:
            Response = await Client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                messages=[
                    {"role": "user", "content": FormattedPrompt}
                ],
                max_tokens=100,
                temperature=0
            )
            
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
    
    # Get unique labels from the dataframe
    ValidLabels = Dataframe['label'].unique().tolist()
    
    # Extract labels from predictions using regex
    ExtractedLabels = []
    for Prediction in PredictionsList:
        ExtractedLabel = ExtractLabelFromPrediction(Prediction, ValidLabels)
        ExtractedLabels.append(ExtractedLabel)
    
    Dataframe['ExtractedLabel'] = ExtractedLabels
    
    # Calculate accuracy using extracted labels
    CorrectPredictions = sum(1 for ExtractedLabel, Label in zip(ExtractedLabels, Dataframe['label']) 
                           if ExtractedLabel is not None and ExtractedLabel.lower() == Label.lower())
    Accuracy = CorrectPredictions / len(Dataframe)
    
    return Accuracy, Dataframe