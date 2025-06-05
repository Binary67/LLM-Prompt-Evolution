import pandas as pd
import os
import asyncio
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

load_dotenv()

async def AnalyzeErrorsAndRevisePrompt(Prompt, Dataframe):
    """
    Analyzes prediction errors and generates a revised prompt to reduce errors.
    
    Args:
        Prompt (str): The original prompt used for predictions
        Dataframe (pd.DataFrame): DataFrame containing 'ModelPrediction', 'ExtractedLabel', and other columns
        
    Returns:
        str: Revised prompt based on error analysis
    """
    
    # Filter dataframe where ExtractedLabel is not equal to ground truth label
    ErrorData = Dataframe[Dataframe['ExtractedLabel'] != Dataframe['label']].copy()
    
    if ErrorData.empty:
        return Prompt  # No errors found, return original prompt
    
    # Initialize Azure OpenAI client
    Client = AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    # Prepare error analysis input
    ErrorSamples = []
    for _, Row in ErrorData.iterrows():
        ErrorSample = f"Text: {Row['text']}\nExpected: {Row['label']}\nPredicted: {Row['ModelPrediction']}\nExtracted: {Row['ExtractedLabel']}"
        ErrorSamples.append(ErrorSample)
    
    ErrorAnalysisInput = "\n\n".join(ErrorSamples[:5])  # Limit to first 5 errors
    
    # First Azure OpenAI call: Analyze errors
    ErrorAnalysisPrompt = f"""
    Analyze the following prediction errors from a text classification model:
    
    Original Prompt: {Prompt}
    
    Error Examples:
    {ErrorAnalysisInput}
    
    Please analyze these errors and identify:
    1. Common patterns in the misclassifications
    2. Potential ambiguities in the original prompt
    3. Missing guidance or instructions that could help the model
    4. Specific areas where the prompt could be clearer
    
    Provide a detailed analysis of what went wrong.
    """
    
    try:
        ErrorAnalysisResponse = await Client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[
                {"role": "user", "content": ErrorAnalysisPrompt}
            ],
        )
        
        ErrorAnalysis = ErrorAnalysisResponse.choices[0].message.content.strip()
        
        # Second Azure OpenAI call: Generate revised prompt
        PromptRevisionRequest = f"""
        Based on the following error analysis, please revise the original prompt to reduce these types of errors:
        
        Original Prompt: {Prompt}
        
        Error Analysis: {ErrorAnalysis}
        
        Please provide a revised prompt that:
        1. Addresses the identified issues
        2. Provides clearer instructions
        3. Includes relevant examples or guidance
        4. Removes any unnecessary, redundant, or outdated instructions from the original prompt. Keeps the prompt concise while maintaining effectiveness
        5. Includes specific samples of wrong predictions together with their correct answers to help the model avoid making the same mistakes
        
        Return only the revised prompt, nothing else.
        """
        
        PromptRevisionResponse = await Client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[
                {"role": "user", "content": PromptRevisionRequest}
            ],
        )
        
        RevisedPrompt = PromptRevisionResponse.choices[0].message.content.strip()
        
        # Remove all {} placeholders and append fixed sentence
        import re
        RevisedPrompt = re.sub(r'\{[^}]*\}', '', RevisedPrompt)
        RevisedPrompt = RevisedPrompt + "\nHere is the talent feedback: {text}"
        
        return RevisedPrompt
        
    except Exception as e:
        print(f"Error in error analysis: {e}")
        return Prompt  # Return original prompt if analysis fails