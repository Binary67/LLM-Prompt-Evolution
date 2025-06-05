import json


def SaveResultsWithBestPrompt(IterationResults, BestPrompt, FilePath='PromptTracing.json'):
    """Save iteration results with the best prompt appended."""
    DataToSave = list(IterationResults)
    DataToSave.append({"BestPromptByF1": BestPrompt})
    with open(FilePath, 'w') as JsonFile:
        json.dump(DataToSave, JsonFile, indent=2)
