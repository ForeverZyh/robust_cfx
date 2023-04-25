import copy
import enum

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# parts copied from https://github.com/junqi-jiang/robust-ce-inn/blob/main/dataset.py

class Datatype(enum.Enum):
    DISCRETE = 0
    ORDINAL = 1
    CONTINUOUS_REAL = 2

CREDIT_FEAT = {
        0: Datatype.DISCRETE, # checking account status
        1: Datatype.ORDINAL, # duration
        2: Datatype.DISCRETE, # credit history
        3: Datatype.DISCRETE, # purpose
        4: Datatype.CONTINUOUS_REAL, # credit amount
        5: Datatype.DISCRETE, # savings
        6: Datatype.DISCRETE, # employment
        7: Datatype.ORDINAL, # installment rate
        8: Datatype.DISCRETE, # personal status
        9: Datatype.DISCRETE, # other debtors
        10: Datatype.CONTINUOUS_REAL, # residence time
        11: Datatype.DISCRETE, # property
        12: Datatype.CONTINUOUS_REAL, # age
        13: Datatype.DISCRETE, # other installment plans
        14: Datatype.DISCRETE, # housing
        15: Datatype.ORDINAL, # number of existing credits
        16: Datatype.DISCRETE, # job
        17: Datatype.ORDINAL, # number of people being liable
        18: Datatype.DISCRETE, # telephone
        19: Datatype.DISCRETE, # foreign worker
    }


class Custom_Dataset(Dataset):
    '''
    Custom dataset class. 
    - data_file is the path of the CSV file containing the data
    - label_col is the name of the label column (if column names are provided), else its index
    - feature_types is a dictionary {(int) feature_index: (Datatype) feature_type}
    '''
    def __init__(self, data_file, label_col, feature_types = CREDIT_FEAT):
        data = pd.read_csv(data_file)
        self.y = data[label_col].values
        self.X = data.drop(columns=[label_col]).values
        self.X = torch.from_numpy(self.X).float()
        self.num_features = self.X.shape[1]

        self.feature_types = feature_types

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]