import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def GenerateData():
    """Create dummy training and validation datasets."""
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
        "Happy with my current work arrangement",
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

    DataTA = DataTA.dropna(subset=['Validation'])

    DataTA['GroundTruth'] = np.where(
        DataTA['Validation'] == 'Agree',
        DataTA['has_aspiration'],
        np.where(DataTA['has_aspiration'] == 'Yes', 'No', 'Yes')
    )

    DataTA = DataTA.rename(columns={'GroundTruth': 'label', 'talent_statement': 'text'})
    DataTA = DataTA[['text', 'label']]
    DataTA = DataTA.replace({'label': {'Yes': 'has_aspiration', 'No': 'no_aspiration'}})

    TrainingData, ValidationData = train_test_split(DataTA, test_size=0.2, stratify=DataTA['label'])
    return TrainingData, ValidationData
