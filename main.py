import pandas as pd
import numpy as np
import asyncio
from sklearn.model_selection import train_test_split
from PromptEvaluation import EvaluatePrompt

async def Main():
    #######################
    ### Data Processing ###
    #######################
    
    # Generate dummy DataTA for testing
    DummyTalentStatements = [
        "I aspire to become a team leader in the next two years",
        "Looking forward to taking on more challenging projects",
        "I want to develop my technical skills further",
        "Seeking opportunities for career advancement",
        "Happy with my current role and responsibilities",
        "Content with maintaining my current position",
        "Not interested in additional responsibilities",
        "Prefer to focus on work-life balance",
        "Eager to learn new technologies and methodologies",
        "Aiming for a promotion within the next year",
        "Would like to mentor junior team members",
        "Planning to pursue additional certifications",
        "Satisfied with current workload and duties",
        "Not looking for career changes at this time",
        "Interested in cross-functional collaboration",
        "Hoping to lead strategic initiatives"
    ]
    
    DummyValidation = ['Agree', 'Agree', 'Agree', 'Agree', 'Disagree', 'Disagree', 'Disagree', 'Disagree',
                       'Agree', 'Agree', 'Agree', 'Agree', 'Disagree', 'Disagree', 'Agree', 'Agree']
    
    DummyHasAspiration = ['Yes', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No', 'No',
                          'Yes', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes']
    
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

    TrainingData, ValidationData = train_test_split(DataTA, test_size = 0.33, stratify = DataTA['label'])

    ########################
    ### Prompt Evaluation ##
    ########################
    ExamplePrompt = "Analyze the following talent feedback and determine if it shows aspiration. Respond with 'has_aspiration' or 'has_aspiration'. {text}"
    
    Accuracy, EvaluationResults = await EvaluatePrompt(ExamplePrompt, ValidationData)
    print(f"Accuracy: {Accuracy:.3f}")

if __name__ == "__main__":
    asyncio.run(Main())