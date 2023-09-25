import copy
import enum
from typing import Union, Optional, Dict, Any, List, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


# parts copied from https://github.com/junqi-jiang/robust-ce-inn/blob/main/dataset.py

class DataType(enum.Enum):
    DISCRETE = 0
    ORDINAL = 1
    CONTINUOUS_REAL = 2


class Immutability(enum.Enum):
    ANY = 0
    INCREASE = 1
    NONE = 2


CREDIT_FEAT = {
    0: DataType.DISCRETE,  # checking account status
    1: DataType.DISCRETE,  # credit history
    2: DataType.DISCRETE,  # purpose
    3: DataType.DISCRETE,  # savings
    4: DataType.DISCRETE,  # employment
    5: DataType.ORDINAL,  # installment rate 
    6: DataType.DISCRETE,  # personal status
    7: DataType.DISCRETE,  # other debtors
    8: DataType.ORDINAL,  # residence time 
    9: DataType.DISCRETE,  # property
    10: DataType.DISCRETE,  # other installment plans
    11: DataType.DISCRETE,  # housing
    12: DataType.ORDINAL,  # number of existing credits
    13: DataType.DISCRETE,  # job
    14: DataType.DISCRETE,  # number of people being liable
    15: DataType.DISCRETE,  # telephone
    16: DataType.DISCRETE,  # foreign worker
    17: DataType.CONTINUOUS_REAL,  # duration
    18: DataType.CONTINUOUS_REAL,  # credit amount
    19: DataType.CONTINUOUS_REAL,  # age
}

# NOTE marking all ordinal features as discrete because proto can only handle one-hot encoding, not ordinal encoding
CREDIT_FEAT_PROTO = {
    0: DataType.DISCRETE,  # checking account status
    1: DataType.DISCRETE,  # credit history
    2: DataType.DISCRETE,  # purpose
    3: DataType.DISCRETE,  # savings
    4: DataType.DISCRETE,  # employment
    5: DataType.DISCRETE,  # installment rate # ORDINAL
    6: DataType.DISCRETE,  # personal status
    7: DataType.DISCRETE,  # other debtors
    8: DataType.DISCRETE,  # residence time # ORDINAL
    9: DataType.DISCRETE,  # property
    10: DataType.DISCRETE,  # other installment plans
    11: DataType.DISCRETE,  # housing
    12: DataType.DISCRETE,  # number of existing credits #ORDINAL
    13: DataType.DISCRETE,  # job
    14: DataType.DISCRETE,  # number of people being liable
    15: DataType.DISCRETE,  # telephone
    16: DataType.DISCRETE,  # foreign worker
    17: DataType.CONTINUOUS_REAL,  # duration
    18: DataType.CONTINUOUS_REAL,  # credit amount
    19: DataType.CONTINUOUS_REAL,  # age
}

HELOC_FEAT = {
    0: DataType.CONTINUOUS_REAL, # ExternalRiskEstimate
    1: DataType.CONTINUOUS_REAL, # MSinceOldestTradeOpen
    2: DataType.CONTINUOUS_REAL, # MSinceMostRecentTradeOpen
    3: DataType.CONTINUOUS_REAL, # AverageMInFile
    4: DataType.CONTINUOUS_REAL, # NumSatisfactoryTrades
    5: DataType.CONTINUOUS_REAL, # NumTrades60Ever2DerogPubRec
    6: DataType.CONTINUOUS_REAL, # NumTrades90Ever2DerogPubRec
    7: DataType.CONTINUOUS_REAL, # PercentTradesNeverDelq
    8: DataType.CONTINUOUS_REAL, # MSinceMostRecentDelq
    9: DataType.CONTINUOUS_REAL, # MaxDelq2PublicRecLast12M
    10: DataType.CONTINUOUS_REAL, # MaxDelqEver
    11: DataType.CONTINUOUS_REAL, # NumTotalTrades
    12: DataType.CONTINUOUS_REAL, # NumTradesOpeninLast12M
    13: DataType.CONTINUOUS_REAL, # PercentInstallTrades
    14: DataType.CONTINUOUS_REAL, # MSinceMostRecentInqexcl7days
    15: DataType.CONTINUOUS_REAL, # NumInqLast6M
    16: DataType.CONTINUOUS_REAL, # NumInqLast6Mexcl7days
    17: DataType.CONTINUOUS_REAL, # NetFractionRevolvingBurden
    18: DataType.CONTINUOUS_REAL, # NetFractionInstallBurden
    19: DataType.CONTINUOUS_REAL, # NumRevolvingTradesWBalance
    20: DataType.CONTINUOUS_REAL, # NumInstallTradesWBalance
    21: DataType.CONTINUOUS_REAL, # NumBank2NatlTradesWHighUtilization
    22: DataType.CONTINUOUS_REAL, # PercentTradesWBalance
}

ORDINAL_FEATURES_CREDIT = {"installment_rate": 4, "present_residence": 4, "number_credits": 4}
DISCRETE_FEATURES_CREDIT = {"status": 4, "credit_history": 5, "purpose": 10, "savings": 5, "employment_duration": 5,
                            "personal_status_sex": 4, "other_debtors": 3, "property": 4, "other_installment_plans": 3,
                            "housing": 3,
                            "job": 4, "people_liable": 2, "telephone": 2, "foreign_worker": 2}
CONTINUOUS_FEATURES_CREDIT = ["duration", "amount", "age"]

# Indicate whether the CREDIT dataset features are mutable.
# We decided that 
#   - personal status is immutable (sex is immutable, and 
#       marital status is technically mutable but not a good recourse)
#   - age can only be increased
#   - foreign worker is immutable 
IMMUTABILITY_CREDIT_FEAT = {
    0: Immutability.ANY,
    1: Immutability.ANY,
    2: Immutability.ANY,
    3: Immutability.ANY,
    4: Immutability.ANY,
    5: Immutability.ANY,
    6: Immutability.ANY,
    7: Immutability.ANY,
    8: Immutability.NONE,
    9: Immutability.ANY,
    10: Immutability.ANY,
    11: Immutability.ANY,
    12: Immutability.INCREASE,
    13: Immutability.ANY,
    14: Immutability.ANY,
    15: Immutability.ANY,
    16: Immutability.ANY,
    17: Immutability.ANY,
    18: Immutability.ANY,
    19: Immutability.NONE,
}

# continuous varialbes must come last
COLUMNS_CREDIT = ['status', 'credit_history', 'purpose', 'savings', 'employment_duration',
                  'installment_rate', 'personal_status_sex', 'other_debtors', 'present_residence', 'property',
                  'other_installment_plans', 'housing', 'number_credits', 'job', 'people_liable', 'telephone',
                  'foreign_worker', 'duration', 'amount', 'age']
VALS_PER_FEATURE_CREDIT = [4, 5, 10, 5, 5, 4, 4, 3, 4, 4, 3, 3, 4, 4, 2, 2, 2, 1, 1, 1]


class Custom_Dataset(Dataset):
    '''
    Custom dataset class. 
    - data_file is the path of the CSV file containing the data
    - label_col is the name of the label column (if column names are provided), else its index
    - feature_types is a dictionary {(int) feature_index: (Datatype) feature_type}
    '''

    def __init__(self, data_file, label_col, feature_types):
        data = pd.read_csv(data_file)
        X = data.drop(columns=[label_col])

        self.X = torch.from_numpy(X.values).float()
        self.y = data[label_col].values

        self.num_features = self.X.shape[1]
        self.columns = X.columns
        self.feature_types = feature_types
        self.discrete_features = {}
        self.ordinal_features = {}
        self.continuous_features = []
        self.feat_var_map = {}

        # feat var map is just a dictionary mapping x --> x for x in [0, 1, num_feat-1] since we use all features
        self.feat_var_map = {i: [i] for i in range(self.num_features)}

        self.build_discrete_features()
        self.build_ordinal_features()
        self.build_continuous_features()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], idx

    def build_discrete_features(self):
        '''
            self.discrete_features is the map x --> number of possible values for x for all discrete features.
        '''
        for idx in self.feature_types.keys():
            val = self.feature_types[idx]
            if val == DataType.DISCRETE:
                self.discrete_features[self.feat_var_map[idx][0]] = max(self.X[:, idx]) + 1

    def build_ordinal_features(self):
        for idx in self.feature_types.keys():
            val = self.feature_types[idx]
            if val == DataType.ORDINAL:
                self.ordinal_features[self.feat_var_map[idx][0]] = self.X[:, idx].max() + 1

    def build_continuous_features(self):
        ''' List of indices of continuous features'''
        for idx in self.feature_types.keys():
            val = self.feature_types[idx]
            if val == DataType.CONTINUOUS_REAL:
                self.continuous_features.append(self.feat_var_map[idx][0])


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
        self.cont_features = []
        self.min_vals = None
        self.max_vals = None

    def _encode_one_feature(self, df, name, feat_num, type):
        '''
            df - data
            i - index of the feature in the current dataframe
            name - name of the feature (e.g., pre-processed feature index)
            feat_num - number of possible values for this feature
            type - ordinal or discrete or continuous
        '''

        # combining the two lines below into one line doesn't work, idk why
        enccolslist = [x for x in self.enc_cols]
        i = enccolslist.index(name)

        enc_idx = list(range(i, i + feat_num))
        df_front = df[self.enc_cols[:i]]
        df_back = df[self.enc_cols[i + 1:]]

        enc = df[self.enc_cols[i]]
        enc = enc[~np.isnan(enc)]  # to avoid nan bugs from pd
        encoded = np.zeros((len(enc), feat_num))

        if type == "ordinal":
            encoded = self._encode_one_feature_ordinal(enc, encoded)
        elif type == "discrete":
            encoded = self._encode_one_feature_discrete(enc, encoded)

        cols = [name + "_" + str(j) for j in range(feat_num)]
        enc_df = pd.DataFrame(data=encoded, columns=cols)

        new_df = pd.concat([df_front, enc_df, df_back], axis=1)
        return new_df, enc_idx, new_df.columns

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
        if type(df) != type(pd.DataFrame()):
            df = pd.DataFrame(df, columns=self.columns)
        self.enc_cols = self.columns  # reset encoded cols
        df_copy = copy.copy(df)
        for (i, name) in enumerate(self.columns):
            if i in self.ordinal.keys():
                df_copy, self.feature_var_map[i], column_names = self._encode_one_feature(df_copy, name,
                                                                                          int(self.ordinal[i].item()),
                                                                                          "ordinal")
            elif i in self.discrete.keys():
                df_copy, self.feature_var_map[i], column_names = self._encode_one_feature(df_copy, name,
                                                                                          int(self.discrete[i].item()),
                                                                                          "discrete")
            else:
                enccolslist = [x for x in self.enc_cols]
                self.feature_var_map[i], column_names = [enccolslist.index(name)], self.enc_cols  # continuous
            self.enc_cols = column_names

        return df_copy, self.feature_var_map

    def encode_one(self, x):
        """
        encode one point
        :param x: numpy array, shaped (x,)
        :return: x_copy: numpy array
        """
        self.enc_cols = self.columns  # reset encoded cols
        xpd = pd.DataFrame(data=x.reshape(1, -1), columns=self.columns)
        return self.encode_df(xpd)

    def inverse_df(self, df):
        df_copy = copy.copy(df)
        return df_copy

    def inverse_one(self, x):
        x_copy = copy.copy(x)
        return x_copy

    def normalize(self, x, hard=False):
        # normalize discrete features of cfx
        for col in self.discrete:
            cat_idx = self.feature_var_map[col][0]
            cat_end_idx = self.feature_var_map[col][0] + self.discrete[col].long().item()
            if hard:
                x[:, cat_idx: cat_end_idx] = F.gumbel_softmax(x[:, cat_idx: cat_end_idx], hard=hard)
            else:
                x[:, cat_idx: cat_end_idx] = F.softmax(x[:, cat_idx: cat_end_idx], dim=-1)

        # normalize ordinal features of cfx
        for col in self.ordinal:
            cat_idx = self.feature_var_map[col][0]
            cat_end_idx = self.feature_var_map[col][0] + self.ordinal[col].long().item()
            if hard:
                tmp_x = F.gumbel_softmax(x[:, cat_idx: cat_end_idx], hard=hard)
            else:
                tmp_x = F.softmax(x[:, cat_idx: cat_end_idx], dim=-1)
            # reverse the tmp_x and cumsum
            x[:, cat_idx: cat_end_idx] = torch.flip(torch.cumsum(torch.flip(tmp_x, dims=[1]), dim=1), dims=[1])

        # normalize continuous features of cfx
        for col in enumerate(self.cont_features):
            x[:, col] = torch.clamp(x[:, col], 0.0, 1.0)
        return x

    def min_max_scale(self, df, min_vals=None, max_vals=None):
        '''
        Copied from https://github.com/junqi-jiang/robust-ce-inn : ~/expnns/preprocessor.py
        '''
        df_copy = copy.copy(df)
        if min_vals is None:
            min_vals = self.min_vals
        if max_vals is None:
            max_vals = self.max_vals
        for i, name in enumerate(self.cont_features):
            min_val = min_vals[i]
            max_val = max_vals[i]
            df_copy[:, name] = (df_copy[:, name] - min_val) / (max_val - min_val)
        return df_copy

    def make_cont_features(self, data, cont_features):
        self.cont_features = [data.feat_var_map[i][0] for i in cont_features]
        self.min_vals = np.min(data.X[:, self.cont_features], axis=0)
        self.max_vals = np.max(data.X[:, self.cont_features], axis=0)


def load_data(data, label, feature_types, preprocessor=None):
    '''
         Load data and preprocess it to be in the correct format for Proto CFX generation
         Adapted from Jiang et al. Uses a OHE for the data.

            :param data: numpy array of shape (num_samples, num_features)
            :param label: numpy array of shape (num_samples, )
            :param feature_types: list of length num_features, each element is a DataType enum
            :param minmax: minmax scaler (use same scaler for train and test data)
    '''
    data = Custom_Dataset(data, label, feature_types)
    cont_features = [i for i in range(data.num_features) if feature_types[i] == DataType.CONTINUOUS_REAL]
    if preprocessor is None:
        preprocessor = Preprocessor(data.ordinal_features, data.discrete_features, data.columns)
        need_make_cont_features = True
    else:
        need_make_cont_features = False
    data.X, data.feat_var_map = preprocessor.encode_df(data.X)
    data.num_features = data.X.shape[1]
    if type(data.X) != type(torch.Tensor()):
        data.X = np.array(data.X)
    if need_make_cont_features:
        preprocessor.make_cont_features(data, cont_features)
    data.X = preprocessor.min_max_scale(data.X)

    return data, preprocessor


def load_data_v1(data, data_test, label, feature_types):
    train_data = Custom_Dataset(data, label, feature_types)
    test_data = Custom_Dataset(data_test, label, feature_types)

    minmax = MinMaxScaler(clip=True)
    train_data.X = minmax.fit_transform(train_data.X)
    test_data.X = minmax.transform(test_data.X)

    return train_data, test_data, minmax
