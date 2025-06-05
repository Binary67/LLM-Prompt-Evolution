import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    #######################
    ### Data Processing ###
    #######################
    DataTA = pd.read_excel('/dbfs/mnt/uat/Franky/inputData/TA_RetrainingData.xlsx')
    DataTA = DataTA.dropna(subset = ['Validation'])

    DataTA['GroundTruth'] = np.where(
        DataTA['Validation'] == 'Agree',
        DataTA['has_aspiration'],
        np.where(DataTA['has_aspiration'] == 'Yes', 'No', 'Yes')
    )

    DataTA = DataTA.rename(columns = {'GroundTruth': 'label', 'talent_statement': 'text'})
    DataTA = DataTA[['text', 'label']]
    DataTA = DataTA.replace({'label': {'Yes': 'has_apiration', 'No': 'no_aspiration'}})

    TrainingData, ValidationData = train_test_split(DataTA, test_size = 0.33, stratify = DataTA['label'])