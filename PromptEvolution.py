import os
import pandas as pd
from openai import AzureOpenAI
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


def ImprovePrompt(
    Prompt: str,
    Accuracy: float,
    ResultsDataFrame: pd.DataFrame,
    LabelColumn: Optional[str] = None
) -> str:
    """
    Analyze error patterns in model predictions and improve the prompt accordingly.
    
    Args:
        Prompt: The original prompt that was used
        Accuracy: The accuracy score achieved (between 0 and 1)
        ResultsDataFrame: DataFrame containing ground truth and predictions
        LabelColumn: Optional name of the label column (if not provided, will look for common names)
    
    Returns:
        An improved version of the prompt
    """
    # Initialize Azure OpenAI client
    Client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    # Identify label column if not provided
    if LabelColumn is None:
        PossibleLabelColumns = [col for col in ResultsDataFrame.columns 
                               if col.lower() in ['label', 'target', 'class', 'category']]
        if PossibleLabelColumns:
            LabelColumn = PossibleLabelColumns[0]
        else:
            # Assume it's any column that's not 'Prediction'
            LabelColumn = [col for col in ResultsDataFrame.columns 
                          if col != 'Prediction'][0]
    
    # Filter incorrect predictions for analysis
    IncorrectPredictions = ResultsDataFrame[
        ResultsDataFrame['Prediction'] != ResultsDataFrame[LabelColumn].astype(str)
    ]
    
    # Prepare error analysis prompt
    ErrorAnalysisPrompt = f"""Analyze the following incorrect predictions and identify error patterns:

Original Prompt Used:
{Prompt}

Accuracy Achieved: {Accuracy:.2%}

Sample of Incorrect Predictions (showing up to 20):
{IncorrectPredictions.head(20).to_string()}

Please analyze:
1. What are the common patterns in the errors?
2. What types of cases is the model getting wrong?
3. What might be causing these misclassifications?
4. Are there specific features or characteristics that lead to errors?

Provide a concise analysis focusing on actionable insights."""

    # Get error pattern analysis
    ErrorAnalysisResponse = Client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=[
            {"role": "system", "content": "You are an expert in analyzing machine learning errors and improving prompts."},
            {"role": "user", "content": ErrorAnalysisPrompt}
        ],
        temperature=0.7
    )
    
    ErrorAnalysis = ErrorAnalysisResponse.choices[0].message.content
    
    # Prepare prompt improvement request
    ImprovementPrompt = f"""Based on the error analysis below, improve the original prompt to reduce these errors:

Original Prompt:
{Prompt}

Current Accuracy: {Accuracy:.2%}

Error Analysis:
{ErrorAnalysis}

Please provide an improved version of the prompt that:
1. Addresses the identified error patterns
2. Provides clearer instructions to reduce misclassifications
3. Maintains the same variable placeholders as the original
4. Is more specific about edge cases or ambiguous situations
5. Ensures the model outputs ONLY the label without any additional text or explanations
   - Add explicit instructions like "Output only the label:" or "Return only one word:"
   - Consider using format constraints like "Your answer must be exactly one of: [list of valid labels]"
6. Consider adding few-shot learning examples if they would help clarify the classification task
   - Include 2-3 representative examples showing correct classifications
   - Focus examples on cases similar to the error patterns identified
   - Format examples clearly to demonstrate the expected input-output relationship
   - Show examples that output ONLY the label, no explanations
7. IMPORTANT: Keep the prompt concise by removing unnecessary or redundant parts
   - Review the entire prompt and remove any redundant instructions
   - Consolidate similar points into single, clear instructions
   - Remove verbose explanations that don't improve accuracy
   - Aim for clarity and brevity - a shorter, clearer prompt often performs better
   - If the prompt already contains examples, evaluate if they're still relevant or need updating

Return ONLY the improved prompt text, without any explanation or additional commentary."""

    # Get improved prompt
    ImprovementResponse = Client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=[
            {"role": "system", "content": "You are an expert prompt engineer focused on improving classification accuracy."},
            {"role": "user", "content": ImprovementPrompt}
        ],
        temperature=0.7
    )
    
    ImprovedPrompt = ImprovementResponse.choices[0].message.content.strip()
    
    return ImprovedPrompt