import copy
import enum

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

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
    def __init__(self, data_file, label_col, feature_types):
        data = pd.read_csv(data_file)
        self.y = data[label_col].values
        self.X = data.drop(columns=[label_col]).values
        self.X = torch.from_numpy(self.X).float()
        self.num_features = self.X.shape[1]

        self.feature_types = feature_types
        # feat var map is just a dictionary mapping x --> x for x in [0, 1, num_feat-1] since we use all features
        self.feat_var_map = {i: i for i in range(self.num_features)}

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

class Preprocessor:
    """
    class for dataframe dataset preprocessing: necessary for CFX generation
    ordinal features: value 0: [1, 0, 0, 0], value 1: [1, 1, 0, 0], value 2: [1, 1, 1, 0]
    discrete features: value 0: [1, 0, 0, 0], value 1: [0, 1, 0, 0], value 2: [0, 0, 1, 0]
    continuous features: (x - min) / (max - min)

    Copied from:
    https://github.com/junqi-jiang/robust-ce-inn : ~/expnns/preprocessor.py
    """

    def __init__(self, ordinal, discrete, columns):
        self.ordinal = ordinal
        self.discrete = discrete
        self.columns = columns
        self.enc_cols = columns
        self.feature_var_map = dict()

    def _encode_one_feature(self, df, name, feat_num, type):
        i = df.columns.get_loc(name)  # current feature index in the updated dataframe
        enc_idx = list(range(i, i + feat_num))
        df_front = df[self.enc_cols[:i]]
        df_back = df[self.enc_cols[i + 1:]]

        enc = df.values[:, i]
        enc = enc[~np.isnan(enc)]   # to avoid nan bugs from pd
        encoded = np.zeros((len(enc), feat_num))

        if type == "ordinal":
            encoded = self._encode_one_feature_ordinal(enc, encoded)
        elif type == "discrete":
            encoded = self._encode_one_feature_discrete(enc, encoded)

        cols = [name + "_" + str(j) for j in range(feat_num)]
        enc_df = pd.DataFrame(data=encoded, columns=cols)

        new_df = pd.concat([df_front, enc_df, df_back], axis=1)
        return new_df, enc_idx

    def _encode_one_feature_ordinal(self, enc, encoded):
        for loc, val in enumerate(enc):
            vals = val + 1
            encoded[int(loc), :int(vals)] = 1
        return encoded

    def _encode_one_feature_discrete(self, enc, encoded):
        for loc, val in enumerate(enc):
            encoded[int(loc), int(val)] = 1
        return encoded

    def encode_df(self, df):
        self.enc_cols = self.columns # reset encoded cols
        df_copy = copy.copy(df)
        for (i, name) in enumerate(self.columns):
            if name in self.ordinal:
                df_copy, self.feature_var_map[i] = self._encode_one_feature(df_copy, name, self.ordinal[name],
                                                                            "ordinal")
            elif name in self.discrete:
                df_copy, self.feature_var_map[i] = self._encode_one_feature(df_copy, name, self.discrete[name],
                                                                            "discrete")
            else:
                idx = df_copy.columns.get_loc(name)
                self.feature_var_map[i] = [idx]  # continuous
            self.enc_cols = list(df_copy.columns)
        return df_copy

    def encode_one(self, x):
        """
        encode one point
        :param x: numpy array, shaped (x,)
        :return: x_copy: numpy array
        """
        self.enc_cols = self.columns # reset encoded cols
        xpd = pd.DataFrame(data=x.reshape(1, -1), columns=self.columns)
        return self.encode_df(xpd)

    def inverse_df(self, df):
        df_copy = copy.copy(df)
        return df_copy

    def inverse_one(self, x):
        x_copy = copy.copy(x)
        return x_copy
    
def min_max_scale(df, continuous, min_vals=None, max_vals=None):
    ''' 
    Copied from https://github.com/junqi-jiang/robust-ce-inn : ~/expnns/preprocessor.py
    '''
    df_copy = copy.copy(df)
    for i, name in enumerate(continuous):
        if min_vals is None:
            min_val = np.min(df_copy[name])
        else:
            min_val = min_vals[i]
        if max_vals is None:
            max_val = np.max(df_copy[name])
        else:
            max_val = max_vals[i]
        df_copy[name] = (df_copy[name] - min_val) / (max_val - min_val)
    return df_copy


def load_data(data, label, feature_types, df_mm = None):
    train_data = Custom_Dataset(data, label, feature_types)

    cont_features = [i for i in range(train_data.num_features) if feature_types[i] == Datatype.CONTINUOUS_REAL]
    ord_features = [i for i in range(train_data.num_features) if feature_types[i] == Datatype.ORDINAL]
    disc_features = [i for i in range(train_data.num_features) if feature_types[i] == Datatype.DISCRETE]

    min_vals = np.min(train_data.X[:, cont_features], axis=0)
    max_vals = np.max(train_data.X[:, cont_features], axis=0)
    if df_mm is None:
        # allow passing df_mm in case test data is different
        df_mm = min_max_scale(train_data.X, cont_features, min_vals, max_vals)
    preprocessor = Preprocessor(ord_features, disc_features, train_data.columns)
    df_enc = preprocessor.encode_df(df_mm)

    # minmax = MinMaxScaler(clip=True)
    # train_data.X = minmax.fit_transform(train_data.X)
    # test_data.X = minmax.transform(test_data.X)


    return df_enc, df_mm #, minmax