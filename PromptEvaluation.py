import pandas as pd
import os
import asyncio
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

load_dotenv()

async def EvaluatePrompt(Prompt, Dataframe):
    Dataframe = Dataframe.reset_index(drop=True)

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
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
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
    
    Tasks = [ProcessRow(Position, Row) for Position, (_, Row) in enumerate(Dataframe.iterrows())]
    Results = await asyncio.gather(*Tasks)
    
    Predictions = [""] * len(Dataframe)
    for Index, Prediction in Results:
        Predictions[Index] = Prediction
    
    Dataframe['ModelPrediction'] = Predictions
    
    CorrectPredictions = sum(1 for Pred, Label in zip(Predictions, Dataframe['label']) 
                           if Pred.lower() == Label.lower())
    Accuracy = CorrectPredictions / len(Dataframe)
    
    return Accuracy, Dataframe