import pandas as pd
import os
import asyncio
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
import time
import re

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

async def AnalyzeErrorsAndRevisePrompt(Prompt, Dataframe, TargetLabels, ConfusionMatrix):
    """
    Analyzes prediction errors and generates a revised prompt to reduce errors.
    
    Args:
        Prompt (str): The original prompt used for predictions
        Dataframe (pd.DataFrame): DataFrame containing 'ModelPrediction', 'ExtractedLabel', and other columns
        TargetLabels (list): List of valid labels that the model should use
        ConfusionMatrix (pd.DataFrame): Confusion matrix of labels vs predictions
        
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
    
    ErrorAnalysisInput = "\n\n".join(ErrorSamples[:20])  # Limit to first 5 errors
    
    # First Azure OpenAI call: Analyze errors
    ErrorAnalysisPrompt = f"""
    You are an expert prompt engineer. Review these prediction errors from a text classification model.

    Original Prompt:
    {Prompt}

    Error Examples:
    {ErrorAnalysisInput}

    Confusion Matrix:
    {ConfusionMatrix}

    Summarize your findings using bullet points organized by predicted vs true label pairs.
    Reference specific mistakes from the examples above and explain potential causes such as ambiguous phrases,
    missing instructions, or label overlap. Conclude with concise bullet points highlighting what to fix in the prompt.
    """
    
    try:
        async def MakeErrorAnalysisCall():
            return await Client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                messages=[
                    {"role": "user", "content": ErrorAnalysisPrompt}
                ],
            )
        
        ErrorAnalysisResponse = await RetryAzureOpenAICall(MakeErrorAnalysisCall)
        ErrorAnalysis = ErrorAnalysisResponse.choices[0].message.content.strip()
        
        # Format the target labels for the prompt
        LabelOptionsStr = ', '.join([f"'{label}'" for label in TargetLabels])
        
        # Second Azure OpenAI call: Generate revised prompt
        PromptRevisionRequest = f"""
        Based on the following error analysis, please revise the original prompt to reduce these types of errors:
        
        Original Prompt: {Prompt}
        
        Error Analysis: {ErrorAnalysis}
        
        CRITICAL REQUIREMENT: The revised prompt MUST only use these exact labels: {LabelOptionsStr}
        DO NOT introduce any new labels or categories that are not in this list. The model should ONLY output one of these labels.
        
        Please provide a revised prompt that:
        1. Addresses the identified issues
        2. Provides clearer instructions
        3. Includes relevant examples with guidance
        4. Removes any unnecessary, redundant, or outdated instructions from the original prompt. Keeps the prompt concise while maintaining effectiveness
        5. Includes specific samples of wrong predictions together with why they are wrong to help the model avoid making the same mistakes
        6. MUST maintain exactly the same classification labels ({LabelOptionsStr}) - do not add new categories like 'uncertain', 'maybe', etc.
        
        Return only the revised prompt, nothing else.
        """
        
        async def MakePromptRevisionCall():
            return await Client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                messages=[
                    {"role": "user", "content": PromptRevisionRequest}
                ],
            )
        
        PromptRevisionResponse = await RetryAzureOpenAICall(MakePromptRevisionCall)
        RevisedPrompt = PromptRevisionResponse.choices[0].message.content.strip()
        
        # Remove all {} placeholders and append fixed sentence
        RevisedPrompt = re.sub(r'\{[^}]*\}', '', RevisedPrompt)
        RevisedPrompt = RevisedPrompt + "\nHere is the talent feedback: {text}"
        
        return RevisedPrompt
        
    except Exception as e:
        print(f"Error in error analysis: {e}")
        return Prompt  # Return original prompt if analysis fails