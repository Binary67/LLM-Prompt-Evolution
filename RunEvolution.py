import asyncio
import pandas as pd
from PromptEvaluation import EvaluatePrompt
from PromptEvolution import AnalyzeErrorsAndRevisePrompt
from PromptSelector import PromptPool, SelectPrompt
from PromptTracingUtils import SaveResultsWithBestPrompt


def GetBestPromptByF1(Results):
    """Return the prompt with the highest F1 score from iteration results."""
    if not Results:
        raise ValueError("No iteration results available")
    BestEntry = max(Results, key=lambda Item: Item["f1"])
    return BestEntry["prompt"]


async def RunEvolution(TrainingData: pd.DataFrame, ValidationData: pd.DataFrame, MaxIterations: int = 5, AccuracyThreshold: float = 0.85, Epsilon: float = 0.1):
    """Execute the iterative prompt improvement process."""
    TargetLabel = ['has_aspiration', 'no_aspiration']
    LabelOptions = ', '.join([f"'{label}'" for label in TargetLabel])
    CurrentPrompt = f"Analyze the following talent feedback and determine if it shows career aspiration. Respond with one of: {LabelOptions} only. Here is the talent feedback: {{text}}"

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

    PromptPoolInstance = PromptPool()
    PromptPoolInstance.AddPrompt(CurrentPrompt, CurrentAccuracy)

    IterationResults.append({
        "iteration": 0,
        "prompt": CurrentPrompt,
        "accuracy": CurrentAccuracy,
        "precision": CurrentPrecision,
        "recall": CurrentRecall,
        "f1": CurrentF1,
    })

    for Iteration in range(MaxIterations):
        print(f"\n--- Iteration {Iteration + 1} ---")

        if CurrentAccuracy >= AccuracyThreshold:
            print(f"Accuracy threshold {AccuracyThreshold:.3f} reached! Stopping iterations.")
            break

        SelectedPrompt = SelectPrompt(PromptPoolInstance, Epsilon)
        print("Evaluating selected prompt...")
        (
            CurrentAccuracy,
            CurrentPrecision,
            CurrentRecall,
            CurrentF1,
            CurrentEvaluationResults,
            CurrentConfusionMatrix,
        ) = await EvaluatePrompt(SelectedPrompt, TrainingData, TargetLabel)

        print("Analyzing errors and generating revised prompt...")
        ErrorCount = len(CurrentEvaluationResults[CurrentEvaluationResults['ExtractedLabel'] != CurrentEvaluationResults['label']])
        print(f"Number of errors found: {ErrorCount}")
        RevisedPrompt = await AnalyzeErrorsAndRevisePrompt(
            SelectedPrompt,
            CurrentEvaluationResults,
            TargetLabel,
            CurrentConfusionMatrix,
        )

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

        CurrentPrompt = RevisedPrompt
        CurrentAccuracy = RevisedAccuracy
        CurrentPrecision = RevisedPrecision
        CurrentRecall = RevisedRecall
        CurrentF1 = RevisedF1
        CurrentEvaluationResults = RevisedEvaluationResults
        CurrentConfusionMatrix = RevisedConfusionMatrix

        PromptPoolInstance.AddPrompt(RevisedPrompt, RevisedAccuracy)

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

    BestPrompt = GetBestPromptByF1(IterationResults)
    print("\nEvaluating best prompt on validation data...")
    (
        ValidationAccuracy,
        ValidationPrecision,
        ValidationRecall,
        ValidationF1,
        _,
        _,
    ) = await EvaluatePrompt(BestPrompt, ValidationData, TargetLabel)
    print(
        f"Validation Accuracy: {ValidationAccuracy:.3f} | Precision: {ValidationPrecision:.3f} | Recall: {ValidationRecall:.3f} | F1: {ValidationF1:.3f}"
    )

    SaveResultsWithBestPrompt(IterationResults, BestPrompt, 'PromptTracing.json')
    print("Results Saved")
