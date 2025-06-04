import pandas as pd
from PromptEvolution import ImprovePrompt
import os
import json
from typing import Dict, List, Tuple
from OpenAiClient import GetClient

def AnalyzeErrorPatterns(ResultsDataFrame: pd.DataFrame, LabelColumn: str) -> Dict[str, List[Dict]]:
    """
    Analyze error patterns from evaluation results.
    
    Args:
        ResultsDataFrame: DataFrame with evaluation results including ExtractedLabel column
        LabelColumn: Name of the column containing true labels
        
    Returns:
        Dictionary containing error patterns categorized by type
    """
    ErrorPatterns = {
        'Misclassifications': [],
        'PatternsByLabel': {},
        'CommonErrors': []
    }
    
    # Find all misclassified samples
    Misclassified = ResultsDataFrame[ResultsDataFrame['ExtractedLabel'] != ResultsDataFrame[LabelColumn].astype(str)]
    
    # Analyze misclassifications
    for _, Row in Misclassified.iterrows():
        ErrorInfo = {
            'Input': Row.to_dict(),
            'TrueLabel': str(Row[LabelColumn]),
            'PredictedLabel': Row['ExtractedLabel'],
            'RawPrediction': Row['Prediction']
        }
        ErrorPatterns['Misclassifications'].append(ErrorInfo)
    
    # Group errors by true label
    for Label in ResultsDataFrame[LabelColumn].unique():
        LabelErrors = Misclassified[Misclassified[LabelColumn] == Label]
        if len(LabelErrors) > 0:
            ErrorPatterns['PatternsByLabel'][str(Label)] = {
                'Count': len(LabelErrors),
                'ErrorRate': len(LabelErrors) / len(ResultsDataFrame[ResultsDataFrame[LabelColumn] == Label]),
                'CommonMisclassifiedAs': LabelErrors['ExtractedLabel'].value_counts().to_dict()
            }
    
    # Find common error types
    if len(Misclassified) > 0:
        # Most confused label pairs
        ConfusionPairs = {}
        for _, Row in Misclassified.iterrows():
            Pair = f"{Row[LabelColumn]} -> {Row['ExtractedLabel']}"
            ConfusionPairs[Pair] = ConfusionPairs.get(Pair, 0) + 1
        
        # Sort by frequency
        SortedPairs = sorted(ConfusionPairs.items(), key=lambda x: x[1], reverse=True)
        ErrorPatterns['CommonErrors'] = [{'Confusion': Pair, 'Count': Count} for Pair, Count in SortedPairs[:5]]
    
    return ErrorPatterns


def CombineErrorFeedback(BestErrorPatterns: Dict, CurrentErrorPatterns: Dict) -> str:
    """
    Combine error patterns from best and current prompts into comprehensive feedback.
    
    Args:
        BestErrorPatterns: Error patterns from the best performing prompt
        CurrentErrorPatterns: Error patterns from the current (failed) prompt
        
    Returns:
        Combined feedback string for prompt improvement
    """
    CombinedFeedback = []
    
    # Add persistent errors from best prompt
    if BestErrorPatterns['CommonErrors']:
        CombinedFeedback.append("Persistent errors from best prompt:")
        for Error in BestErrorPatterns['CommonErrors'][:3]:
            CombinedFeedback.append(f"- {Error['Confusion']}: {Error['Count']} times")
    
    # Add new error patterns from current prompt
    if CurrentErrorPatterns['CommonErrors']:
        CombinedFeedback.append("\nNew error patterns from current attempt:")
        CurrentErrors = {Error['Confusion'] for Error in CurrentErrorPatterns['CommonErrors']}
        BestErrors = {Error['Confusion'] for Error in BestErrorPatterns['CommonErrors']}
        NewErrors = CurrentErrors - BestErrors
        
        if NewErrors:
            for Error in CurrentErrorPatterns['CommonErrors']:
                if Error['Confusion'] in NewErrors:
                    CombinedFeedback.append(f"- {Error['Confusion']}: {Error['Count']} times")
    
    # Add insights about what made current prompt fail
    CombinedFeedback.append("\nKey differences in error patterns:")
    
    # Compare error rates by label
    for Label in set(list(BestErrorPatterns['PatternsByLabel'].keys()) + 
                     list(CurrentErrorPatterns['PatternsByLabel'].keys())):
        BestErrorRate = BestErrorPatterns['PatternsByLabel'].get(Label, {}).get('ErrorRate', 0)
        CurrentErrorRate = CurrentErrorPatterns['PatternsByLabel'].get(Label, {}).get('ErrorRate', 0)
        
        if CurrentErrorRate > BestErrorRate:
            CombinedFeedback.append(f"- '{Label}' classification degraded: {BestErrorRate:.1%} -> {CurrentErrorRate:.1%}")
    
    return "\n".join(CombinedFeedback)


def HybridImprovePrompt(BestPrompt: str, BestAccuracy: float, BestResults: pd.DataFrame,
                        CurrentPrompt: str, CurrentAccuracy: float, CurrentResults: pd.DataFrame,
                        LabelColumn: str) -> str:
    """
    Improve prompt using hybrid feedback from both best and current prompts.
    
    Args:
        BestPrompt: The best performing prompt so far
        BestAccuracy: Accuracy of the best prompt
        BestResults: Evaluation results from the best prompt
        CurrentPrompt: The current prompt that performed worse
        CurrentAccuracy: Accuracy of the current prompt
        CurrentResults: Evaluation results from the current prompt
        LabelColumn: Name of the column containing true labels
        
    Returns:
        Improved prompt incorporating feedback from both attempts
    """
    # Analyze error patterns from both prompts
    BestErrorPatterns = AnalyzeErrorPatterns(BestResults, LabelColumn)
    CurrentErrorPatterns = AnalyzeErrorPatterns(CurrentResults, LabelColumn)
    
    # Combine feedback
    CombinedFeedback = CombineErrorFeedback(BestErrorPatterns, CurrentErrorPatterns)
    
    # Create context for improvement
    ImprovementContext = f"""
You need to improve a prompt for classification. Here's the context:

BEST PROMPT (Accuracy: {BestAccuracy:.2%}):
{BestPrompt}

ATTEMPTED PROMPT (Accuracy: {CurrentAccuracy:.2%}):
{CurrentPrompt}

The attempted prompt performed worse than the best prompt. Here's the combined error analysis:

{CombinedFeedback}

Key insights:
1. The best prompt still has persistent errors that need addressing
2. The attempted prompt introduced new errors while trying to fix existing ones
3. We need to maintain what works in the best prompt while carefully addressing its weaknesses

Based on this analysis, improve the BEST prompt to:
1. Address the persistent error patterns without breaking what already works
2. Avoid the mistakes that made the attempted prompt perform worse
3. Be more precise in distinguishing between commonly confused labels

Return only the improved prompt text.
"""
    Client = GetClient()
    
    Response = Client.chat.completions.create(
        model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=[
            {"role": "system", "content": "You are an expert in prompt engineering. Generate only the improved prompt without explanations."},
            {"role": "user", "content": ImprovementContext}
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    ImprovedPrompt = Response.choices[0].message.content.strip()
    
    return ImprovedPrompt


def GetDetailedErrorAnalysis(BestResults: pd.DataFrame, CurrentResults: pd.DataFrame, 
                             LabelColumn: str, FeatureColumns: List[str]) -> str:
    """
    Provide detailed error analysis comparing best and current results.
    
    Args:
        BestResults: Results from best prompt
        CurrentResults: Results from current prompt
        LabelColumn: Label column name
        FeatureColumns: List of feature columns
        
    Returns:
        Detailed analysis string
    """
    Analysis = []
    
    # Find samples that best prompt got right but current prompt got wrong
    BestCorrect = BestResults[BestResults['ExtractedLabel'] == BestResults[LabelColumn].astype(str)]
    CurrentWrong = CurrentResults[CurrentResults['ExtractedLabel'] != CurrentResults[LabelColumn].astype(str)]
    
    # Find overlapping indices
    DegradedSamples = BestCorrect.index.intersection(CurrentWrong.index)
    
    if len(DegradedSamples) > 0:
        Analysis.append(f"Found {len(DegradedSamples)} samples that degraded from correct to incorrect")
        
        # Show a few examples
        for Idx in list(DegradedSamples)[:3]:
            BestRow = BestResults.loc[Idx]
            CurrentRow = CurrentResults.loc[Idx]
            
            Analysis.append(f"\nExample degradation:")
            for Col in FeatureColumns:
                Analysis.append(f"  {Col}: {BestRow[Col]}")
            Analysis.append(f"  True label: {BestRow[LabelColumn]}")
            Analysis.append(f"  Best prompt predicted: {BestRow['ExtractedLabel']} ✓")
            Analysis.append(f"  Current prompt predicted: {CurrentRow['ExtractedLabel']} ✗")
    
    return "\n".join(Analysis)