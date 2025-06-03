import pandas as pd
import re
from typing import List, Tuple
from OutputGeneration import GenerateOutput


def ExtractLabelFromOutput(Output: str, UniqueLabels: List[str]) -> str:
    """
    Extract a label from GPT output using regex matching.
    
    Args:
        Output: The GPT-generated output text
        UniqueLabels: List of possible label values to match
        
    Returns:
        Matched label or empty string if no match found
    """
    # Escape special regex characters in labels
    EscapedLabels = [re.escape(str(Label)) for Label in UniqueLabels]
    
    # Create pattern that matches any of the labels (case-insensitive)
    Pattern = r'\b(' + '|'.join(EscapedLabels) + r')\b'
    
    # Search for the pattern in the output
    Match = re.search(Pattern, Output, re.IGNORECASE)
    
    if Match:
        # Find which original label matches (case-insensitive)
        MatchedText = Match.group(1)
        for Label in UniqueLabels:
            if str(Label).lower() == MatchedText.lower():
                return str(Label)
    
    return ""


def EvaluatePrompt(
    Prompt: str, 
    DataFrame: pd.DataFrame, 
    FeatureColumns: List[str], 
    LabelColumn: str
) -> Tuple[float, pd.DataFrame]:
    """
    Evaluate a prompt by using it to predict labels and calculating accuracy.
    
    Args:
        Prompt: The prompt template with placeholders for features
        DataFrame: The dataframe to evaluate on
        FeatureColumns: List of column names to use as features
        LabelColumn: The column name containing true labels
    
    Returns:
        Tuple containing:
        - Accuracy score (float between 0 and 1)
        - DataFrame with additional 'Prediction' column
    """
    # Get unique labels from the label column
    UniqueLabels = DataFrame[LabelColumn].unique().tolist()
    UniqueLabels = [str(Label) for Label in UniqueLabels]
    
    Predictions = []
    ExtractedLabels = []
    
    # Generate predictions for each row
    for _, Row in DataFrame.iterrows():
        # Create variables dict for the prompt
        Variables = {Col: Row[Col] for Col in FeatureColumns}
        
        # Generate prediction using the prompt
        Prediction = GenerateOutput(Prompt, **Variables)
        Predictions.append(Prediction.strip())
        
        # Extract label from prediction
        ExtractedLabel = ExtractLabelFromOutput(Prediction, UniqueLabels)
        ExtractedLabels.append(ExtractedLabel)
    
    # Add predictions to dataframe
    ResultDataFrame = DataFrame.copy()
    ResultDataFrame['Prediction'] = Predictions
    ResultDataFrame['ExtractedLabel'] = ExtractedLabels
    
    # Calculate accuracy using extracted labels
    CorrectPredictions = sum(
        ResultDataFrame['ExtractedLabel'] == ResultDataFrame[LabelColumn].astype(str)
    )
    Accuracy = CorrectPredictions / len(ResultDataFrame)
    
    return Accuracy, ResultDataFrame