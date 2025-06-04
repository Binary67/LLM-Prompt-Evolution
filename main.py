import pandas as pd
import asyncio
from PromptEvaluation import EvaluatePromptAsync
from PromptEvolution import ImprovePrompt
from HybridPromptEvolution import HybridImprovePrompt
from sklearn.model_selection import train_test_split
import numpy as np

async def Main(DataFrame, FeatureColumns, LabelColumn, PromptTemplate, MaxIterations=5, AccuracyThreshold=0.95, BatchSize=10, UseFewShot=False):
    
    # Evaluate the prompt
    print("Evaluating Prompt on Training Data")
    print(f"Number of samples: {len(DataFrame)}")
    
    Accuracy, ResultDataFrame = await EvaluatePromptAsync(
        Prompt=PromptTemplate,
        DataFrame=DataFrame,
        FeatureColumns=FeatureColumns,
        LabelColumn=LabelColumn,
        BatchSize=BatchSize
    )
    
    # Display results
    print(f"Accuracy: {Accuracy:.2%}")
    
    # Summary statistics
    print("\nSummary Statistics:")
    print("-" * 80)
    CorrectCount = sum(ResultDataFrame['ExtractedLabel'] == ResultDataFrame[LabelColumn].astype(str))
    print(f"Correct Predictions: {CorrectCount}/{len(ResultDataFrame)}")
    print(f"Accuracy: {Accuracy:.2%}")
    
    # Iterative prompt improvement
    CurrentPrompt = PromptTemplate
    CurrentAccuracy = Accuracy
    CurrentResults = ResultDataFrame
    IterationHistory = []
    
    # Track the best prompt and its accuracy
    BestPrompt = CurrentPrompt
    BestAccuracy = CurrentAccuracy
    BestResults = CurrentResults
    BestIteration = 0
    
    # Store initial results
    IterationHistory.append({
        'Iteration': 0,
        'Accuracy': CurrentAccuracy,
        'Prompt': CurrentPrompt
    })
    
    print("\n" + "=" * 80)
    print("ITERATIVE PROMPT IMPROVEMENT")
    print("=" * 80)
    print(f"Max iterations: {MaxIterations}")
    print(f"Target accuracy: {AccuracyThreshold:.2%}")
    print(f"Initial accuracy: {CurrentAccuracy:.2%}")
    
    # Run improvement loop
    for Iteration in range(1, MaxIterations + 1):
        # Check if we've reached the accuracy threshold
        if CurrentAccuracy >= AccuracyThreshold:
            print(f"\nTarget accuracy of {AccuracyThreshold:.2%} achieved!")
            break
            
        print(f"\n{'='*80}")
        print(f"ITERATION {Iteration}")
        print(f"{'='*80}")
        print(f"Current accuracy: {CurrentAccuracy:.2%}")
        print(f"Best accuracy so far: {BestAccuracy:.2%}")
        
        # Decide which prompt to improve
        if CurrentAccuracy < BestAccuracy:
            print("Current prompt is worse than best. Using hybrid approach to learn from both prompts.")
            # Use hybrid approach to combine feedback from both prompts

            ImprovedPrompt = HybridImprovePrompt(
                BestPrompt=BestPrompt,
                BestAccuracy=BestAccuracy,
                BestResults=BestResults,
                CurrentPrompt=CurrentPrompt,
                CurrentAccuracy=CurrentAccuracy,
                CurrentResults=CurrentResults,
                LabelColumn=LabelColumn,
                IncludeFewShotExamples=UseFewShot
            )
        else:
            print("Current prompt is performing well. Continuing to improve current prompt.")
            # Use regular improvement for successful prompts
            ImprovedPrompt = ImprovePrompt(
                Prompt=CurrentPrompt,
                Accuracy=CurrentAccuracy,
                ResultsDataFrame=CurrentResults,
                LabelColumn=LabelColumn,
                IncludeFewShotExamples=UseFewShot
            )
        
        print("\nImproved Prompt:")
        print("-" * 80)
        print(ImprovedPrompt)
        print("-" * 80)
        
        # Evaluate the improved prompt
        print("\nEvaluating Improved Prompt")
        ImprovedAccuracy, ImprovedResults = await EvaluatePromptAsync(
            Prompt=ImprovedPrompt,
            DataFrame=DataFrame,
            FeatureColumns=FeatureColumns,
            LabelColumn=LabelColumn,
            BatchSize=BatchSize
        )
        
        # Display iteration results
        print(f"\nIteration {Iteration} Results:")
        print(f"  Previous Accuracy: {CurrentAccuracy:.2%}")
        print(f"  New Accuracy: {ImprovedAccuracy:.2%}")
        print(f"  Improvement: {(ImprovedAccuracy - CurrentAccuracy):.2%}")
        
        # Store iteration results
        IterationHistory.append({
            'Iteration': Iteration,
            'Accuracy': ImprovedAccuracy,
            'Prompt': ImprovedPrompt
        })
        
        # Check if this is the best prompt so far
        if ImprovedAccuracy > BestAccuracy:
            BestPrompt = ImprovedPrompt
            BestAccuracy = ImprovedAccuracy
            BestResults = ImprovedResults
            BestIteration = Iteration
            print(f"\nNew best prompt found! Accuracy: {BestAccuracy:.2%}")
        
        # Update current values
        CurrentPrompt = ImprovedPrompt
        CurrentAccuracy = ImprovedAccuracy
        CurrentResults = ImprovedResults
    
    # Display final summary
    print("\n" + "=" * 80)
    print("IMPROVEMENT SUMMARY")
    print("=" * 80)
    print(f"\nTotal iterations: {len(IterationHistory) - 1}")
    print("\nAccuracy progression:")
    for Entry in IterationHistory:
        print(f"  Iteration {Entry['Iteration']}: {Entry['Accuracy']:.2%}")
    
    print(f"\nTotal improvement: {(BestAccuracy - Accuracy):.2%}")
    print(f"Final accuracy: {CurrentAccuracy:.2%}")
    print(f"Best accuracy: {BestAccuracy:.2%} (achieved at iteration {BestIteration})")

    
    # Save best prompt to file
    with open('BestPrompt.txt', 'w') as File:
        File.write(BestPrompt)
    print(f"Best prompt saved (from iteration {BestIteration})")
    
    return BestPrompt, BestAccuracy


async def TestBestPromptOnValidation(BestPrompt, ValidationData, FeatureColumns, LabelColumn, BatchSize=10):
    """
    Test the best prompt on validation data and return accuracy and dataframe with predictions.
    
    Args:
        BestPrompt: The best prompt obtained from training
        ValidationData: The validation dataframe
        FeatureColumns: List of column names to use as features
        LabelColumn: The column name containing true labels
    
    Returns:
        Tuple containing:
        - Accuracy score (float between 0 and 1)
        - DataFrame with additional 'Prediction' and 'ExtractedLabel' columns
    """
    print("\n" + "=" * 80)
    print("TESTING BEST PROMPT ON VALIDATION DATA")
    print("=" * 80)
    print(f"Number of validation samples: {len(ValidationData)}")
    
    # Evaluate the best prompt on validation data
    Accuracy, ResultDataFrame = await EvaluatePromptAsync(
        Prompt=BestPrompt,
        DataFrame=ValidationData,
        FeatureColumns=FeatureColumns,
        LabelColumn=LabelColumn,
        BatchSize=BatchSize
    )
    
    # Display results
    print(f"\nValidation Accuracy: {Accuracy:.2%}")
    print("\nValidation Results Summary:")
    print("-" * 80)
    
    CorrectCount = sum(ResultDataFrame['ExtractedLabel'] == ResultDataFrame[LabelColumn].astype(str))
    print(f"Correct Predictions: {CorrectCount}/{len(ResultDataFrame)}")
    
    # Display confusion matrix style summary
    UniqueLabels = ResultDataFrame[LabelColumn].unique()
    print("\nPer-Label Performance:")
    for Label in UniqueLabels:
        LabelMask = ResultDataFrame[LabelColumn] == Label
        LabelCorrect = sum((ResultDataFrame[LabelColumn] == Label) & 
                          (ResultDataFrame['ExtractedLabel'] == str(Label)))
        LabelTotal = sum(LabelMask)
        LabelAccuracy = LabelCorrect / LabelTotal if LabelTotal > 0 else 0
        print(f"  {Label}: {LabelCorrect}/{LabelTotal} ({LabelAccuracy:.2%})")
    
    return Accuracy, ResultDataFrame


async def RunMain():
    DataTA = pd.read_excel('/home/frank/Projects/Projects/LLM-Prompt-Evolution/TA_RetrainingData.xlsx')
    DataTA = DataTA.dropna(subset = ['Validation'])

    DataTA['GroundTruth'] = np.where(
        DataTA['Validation'] == 'Agree',
        DataTA['has_aspiration'],
        np.where(DataTA['has_aspiration'] == 'Yes', 'No', 'Yes')
    )

    DataTA = DataTA.rename(columns = {'GroundTruth': 'label', 'talent_statement': 'text'})
    DataTA = DataTA[['text', 'label']]
    DataTA = DataTA.replace({'label': {'Yes': 'has_aspiration', 'No': 'no_aspiration'}})

    TrainingData, ValidationData = train_test_split(DataTA, test_size = 0.33, stratify = DataTA['label'])

    # Define feature columns and label column
    FeatureColumns = ['text']
    LabelColumn = 'label'
    
    # Create prompt template for talent aspiration detection
    PromptTemplate = """Analyze the following talent statement and determine if it expresses career aspiration.

Talent Statement: {text}

Does this statement express career aspiration? Please respond with only: has_aspiration or no_aspiration."""
    
    BestPrompt, BestAccuracy = await Main(
        DataFrame=TrainingData,
        FeatureColumns=FeatureColumns,
        LabelColumn=LabelColumn,
        PromptTemplate=PromptTemplate,
        MaxIterations=5,
        AccuracyThreshold=0.95,
        BatchSize=10,
        UseFewShot=False
    )
    
    # Test the best prompt on validation data
    ValidationAccuracy, ValidationResults = await TestBestPromptOnValidation(
        BestPrompt=BestPrompt,
        ValidationData=ValidationData,
        FeatureColumns=FeatureColumns,
        LabelColumn=LabelColumn,
        BatchSize=10
    )


if __name__ == "__main__":
    asyncio.run(RunMain())