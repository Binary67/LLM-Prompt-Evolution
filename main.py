import pandas as pd
import numpy as np
import asyncio
import json
from sklearn.model_selection import train_test_split
from PromptEvaluation import EvaluatePrompt
from PromptEvolution import AnalyzeErrorsAndRevisePrompt

async def Main(MaxIterations=5, AccuracyThreshold=0.8):
    #######################
    ### Data Processing ###
    #######################
    
    # Generate dummy DataTA for testing
    DummyTalentStatements = [
        # Has aspiration examples
        "I aspire to become a team leader in the next two years",
        "Looking forward to taking on more challenging projects",
        "I want to develop my technical skills further",
        "Seeking opportunities for career advancement",
        "Eager to learn new technologies and methodologies",
        "Aiming for a promotion within the next year",
        "Would like to mentor junior team members",
        "Planning to pursue additional certifications",
        "Interested in cross-functional collaboration",
        "Hoping to lead strategic initiatives",
        "Want to expand my role to include strategic planning",
        "Interested in transitioning to management",
        "Looking to specialize in emerging technologies",
        "Aspire to become a subject matter expert",
        "Seeking international assignment opportunities",
        "Want to contribute to company innovation projects",
        "Planning to pursue an MBA to advance my career",
        "Interested in leading a product development team",
        "Aiming to become a technical architect",
        "Looking for opportunities to present at conferences",
        # No aspiration examples
        "Happy with my current role and responsibilities",
        "Content with maintaining my current position",
        "Not interested in additional responsibilities",
        "Prefer to focus on work-life balance",
        "Satisfied with current workload and duties",
        "Not looking for career changes at this time",
        "Comfortable in my current position",
        "Prefer to stay in my area of expertise",
        "Not seeking additional challenges at this time",
        "Content with my current level of responsibility",
        "Focused on maintaining stability in my role",
        "Happy to continue in my current capacity",
        "Not interested in management responsibilities",
        "Prefer individual contributor role",
        "Satisfied with current career trajectory",
        "Not looking to change my work situation",
        "Content with my current team and duties",
        "Prefer predictable work responsibilities",
        "Not interested in leadership positions",
        "Happy with my current work arrangement"
    ]
    
    DummyValidation = [
        # Mixed validations for has aspiration examples
        'Agree', 'Agree', 'Agree', 'Agree', 'Agree', 'Agree', 'Agree', 'Agree', 
        'Agree', 'Agree', 'Agree', 'Agree', 'Agree', 'Agree', 'Agree', 'Disagree',
        'Disagree', 'Disagree', 'Disagree', 'Disagree',
        # Mixed validations for no aspiration examples
        'Agree', 'Agree', 'Agree', 'Agree', 'Agree', 'Agree', 'Agree', 'Agree',
        'Agree', 'Agree', 'Agree', 'Agree', 'Agree', 'Agree', 'Agree', 'Disagree',
        'Disagree', 'Disagree', 'Disagree', 'Disagree'
    ]
    
    DummyHasAspiration = [
        # Has aspiration labels
        'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes',
        'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes',
        'Yes', 'Yes', 'Yes', 'Yes',
        # No aspiration labels
        'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No',
        'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No',
        'No', 'No', 'No', 'No'
    ]
    
    DataTA = pd.DataFrame({
        'talent_statement': DummyTalentStatements,
        'Validation': DummyValidation,
        'has_aspiration': DummyHasAspiration
    })
    
    DataTA = DataTA.dropna(subset = ['Validation'])

    DataTA['GroundTruth'] = np.where(
        DataTA['Validation'] == 'Agree',
        DataTA['has_aspiration'],
        np.where(DataTA['has_aspiration'] == 'Yes', 'No', 'Yes')
    )

    DataTA = DataTA.rename(columns = {'GroundTruth': 'label', 'talent_statement': 'text'})
    DataTA = DataTA[['text', 'label']]
    DataTA = DataTA.replace({'label': {'Yes': 'has_aspiration', 'No': 'no_aspiration'}})

    TrainingData, ValidationData = train_test_split(DataTA, test_size = 0.2, stratify = DataTA['label'])

    ########################
    ### Prompt Evaluation ##
    ########################
    TargetLabel = ['has_aspiration', 'no_aspiration']
    LabelOptions = ', '.join([f"'{label}'" for label in TargetLabel])
    CurrentPrompt = f"Analyze the following talent feedback and determine if it shows career aspiration. Respond with one of: {LabelOptions} only. Here is the talent feedback: {{text}}"
    
    # Initialize results list to store prompt and accuracy for each iteration
    IterationResults = []
    
    (
        CurrentAccuracy,
        CurrentPrecision,
        CurrentRecall,
        CurrentF1,
        CurrentEvaluationResults,
        CurrentConfusionMatrix,
    ) = await EvaluatePrompt(CurrentPrompt, TrainingData, TargetLabel)
    print(
        f"Initial Accuracy: {CurrentAccuracy:.3f} | Precision: {CurrentPrecision:.3f} | Recall: {CurrentRecall:.3f} | F1: {CurrentF1:.3f}"
    )
    
    # Store initial prompt and accuracy
    IterationResults.append({
        "iteration": 0,
        "prompt": CurrentPrompt,
        "accuracy": CurrentAccuracy,
        "precision": CurrentPrecision,
        "recall": CurrentRecall,
        "f1": CurrentF1,
    })
    
    #########################
    ### Iterative Improvement ###
    #########################
    
    for Iteration in range(MaxIterations):
        print(f"\n--- Iteration {Iteration + 1} ---")
        
        # Check if accuracy threshold is reached
        if CurrentAccuracy >= AccuracyThreshold:
            print(f"Accuracy threshold {AccuracyThreshold:.3f} reached! Stopping iterations.")
            break
            
        # Analyze errors and get revised prompt
        print("Analyzing errors and generating revised prompt...")
        print(f"Number of evaluation results: {len(CurrentEvaluationResults)}")
        ErrorCount = len(CurrentEvaluationResults[CurrentEvaluationResults['ExtractedLabel'] != CurrentEvaluationResults['label']])
        print(f"Number of errors found: {ErrorCount}")
        RevisedPrompt = await AnalyzeErrorsAndRevisePrompt(
            CurrentPrompt,
            CurrentEvaluationResults,
            TargetLabel,
            CurrentConfusionMatrix,
        )
        print(f"Revised Prompt: {RevisedPrompt}")
        
        # Evaluate revised prompt
        print("Evaluating revised prompt...")
        (
            RevisedAccuracy,
            RevisedPrecision,
            RevisedRecall,
            RevisedF1,
            RevisedEvaluationResults,
            RevisedConfusionMatrix,
        ) = await EvaluatePrompt(RevisedPrompt, TrainingData, TargetLabel)
        print(
            f"Revised Accuracy: {RevisedAccuracy:.3f} | Precision: {RevisedPrecision:.3f} | Recall: {RevisedRecall:.3f} | F1: {RevisedF1:.3f}"
        )
        print(f"Improvement: {RevisedAccuracy - CurrentAccuracy:.3f}")
        
        # Update current prompt and accuracy for next iteration
        CurrentPrompt = RevisedPrompt
        CurrentAccuracy = RevisedAccuracy
        CurrentPrecision = RevisedPrecision
        CurrentRecall = RevisedRecall
        CurrentF1 = RevisedF1
        CurrentEvaluationResults = RevisedEvaluationResults
        CurrentConfusionMatrix = RevisedConfusionMatrix
        
        # Store iteration results
        IterationResults.append({
            "iteration": Iteration + 1,
            "prompt": CurrentPrompt,
            "accuracy": CurrentAccuracy,
            "precision": CurrentPrecision,
            "recall": CurrentRecall,
            "f1": CurrentF1,
        })
    
    print(
        f"\nFinal Accuracy after {min(Iteration + 1, MaxIterations)} iterations: {CurrentAccuracy:.3f} | Precision: {CurrentPrecision:.3f} | Recall: {CurrentRecall:.3f} | F1: {CurrentF1:.3f}"
    )
    
    # Save results to JSON file
    with open('PromptTracing.json', 'w') as JsonFile:
        json.dump(IterationResults, JsonFile, indent=2)
    print(f"Results Saved")

if __name__ == "__main__":
    asyncio.run(Main())