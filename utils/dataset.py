import copy
import enum

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

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

HELOC_FEAT = {
    0: DataType.CONTINUOUS_REAL,  # ExternalRiskEstimate
    1: DataType.CONTINUOUS_REAL,  # MSinceOldestTradeOpen
    2: DataType.CONTINUOUS_REAL,  # MSinceMostRecentTradeOpen
    3: DataType.CONTINUOUS_REAL,  # AverageMInFile
    4: DataType.CONTINUOUS_REAL,  # NumSatisfactoryTrades
    5: DataType.CONTINUOUS_REAL,  # NumTrades60Ever2DerogPubRec
    6: DataType.CONTINUOUS_REAL,  # NumTrades90Ever2DerogPubRec
    7: DataType.CONTINUOUS_REAL,  # PercentTradesNeverDelq
    8: DataType.CONTINUOUS_REAL,  # MSinceMostRecentDelq
    9: DataType.CONTINUOUS_REAL,  # MaxDelq2PublicRecLast12M
    10: DataType.CONTINUOUS_REAL,  # MaxDelqEver
    11: DataType.CONTINUOUS_REAL,  # NumTotalTrades
    12: DataType.CONTINUOUS_REAL,  # NumTradesOpeninLast12M
    13: DataType.CONTINUOUS_REAL,  # PercentInstallTrades
    14: DataType.CONTINUOUS_REAL,  # MSinceMostRecentInqexcl7days
    15: DataType.CONTINUOUS_REAL,  # NumInqLast6M
    16: DataType.CONTINUOUS_REAL,  # NumInqLast6Mexcl7days
    17: DataType.CONTINUOUS_REAL,  # NetFractionRevolvingBurden
    18: DataType.CONTINUOUS_REAL,  # NetFractionInstallBurden
    19: DataType.CONTINUOUS_REAL,  # NumRevolvingTradesWBalance
    20: DataType.CONTINUOUS_REAL,  # NumInstallTradesWBalance
    21: DataType.CONTINUOUS_REAL,  # NumBank2NatlTradesWHighUtilization
    22: DataType.CONTINUOUS_REAL,  # PercentTradesWBalance
}

CTG_FEAT = {
    0: DataType.CONTINUOUS_REAL,  # LB
    1: DataType.CONTINUOUS_REAL,  # AC
    2: DataType.CONTINUOUS_REAL,  # FM
    3: DataType.CONTINUOUS_REAL,  # UC
    4: DataType.CONTINUOUS_REAL,  # DL
    5: DataType.CONTINUOUS_REAL,  # DS
    6: DataType.CONTINUOUS_REAL,  # DP
    7: DataType.CONTINUOUS_REAL,  # ASTV
    8: DataType.CONTINUOUS_REAL,  # MSTV
    9: DataType.CONTINUOUS_REAL,  # ALTV
    10: DataType.CONTINUOUS_REAL,  # MLTV
    11: DataType.CONTINUOUS_REAL,  # Width
    12: DataType.CONTINUOUS_REAL,  # Min
    13: DataType.CONTINUOUS_REAL,  # Max
    14: DataType.CONTINUOUS_REAL,  # Nmax
    15: DataType.CONTINUOUS_REAL,  # Nzeros
    16: DataType.CONTINUOUS_REAL,  # Mode
    17: DataType.CONTINUOUS_REAL,  # Mean
    18: DataType.CONTINUOUS_REAL,  # Median
    19: DataType.CONTINUOUS_REAL,  # Variance
    20: DataType.CONTINUOUS_REAL,  # Tendency
}

STUDENT_FEAT = {
    0: DataType.CONTINUOUS_REAL,  # num_prev_attempts
    1: DataType.CONTINUOUS_REAL,  # weight
    2: DataType.CONTINUOUS_REAL,  # weighted_score
    3: DataType.CONTINUOUS_REAL,  # forumng_click
    4: DataType.CONTINUOUS_REAL,  # homepage_click
    5: DataType.CONTINUOUS_REAL,  # oucontent_click
    6: DataType.CONTINUOUS_REAL,  # resource_click
    7: DataType.CONTINUOUS_REAL,  # subpage_click
    8: DataType.CONTINUOUS_REAL,  # url click
    9: DataType.CONTINUOUS_REAL,  # dataplus_click
    10: DataType.CONTINUOUS_REAL,  # glossary_click
    11: DataType.CONTINUOUS_REAL,  # ou_collaborate_click
    12: DataType.CONTINUOUS_REAL,  # quiz_click
    13: DataType.CONTINUOUS_REAL,  # ouelluminate_click
    14: DataType.CONTINUOUS_REAL,  # sharedsubpage_click
    15: DataType.CONTINUOUS_REAL,  # questionnaire_click
    16: DataType.CONTINUOUS_REAL,  # page_click
    17: DataType.CONTINUOUS_REAL,  # externalquiz_click
    18: DataType.CONTINUOUS_REAL,  # ouwiki_click
    19: DataType.CONTINUOUS_REAL,  # dualpane_click
    20: DataType.CONTINUOUS_REAL,  # folder_click
    21: DataType.CONTINUOUS_REAL,  # repeatactivity_click
    22: DataType.CONTINUOUS_REAL,  # htmlactivity_click
    23: DataType.DISCRETE,  # code_module
    24: DataType.DISCRETE,  # gender
    25: DataType.DISCRETE,  # region
    26: DataType.DISCRETE,  # highest_education
    27: DataType.DISCRETE,  # imd_band
    28: DataType.DISCRETE,  # age_band
    29: DataType.DISCRETE,  # studied_credits
    30: DataType.DISCRETE,  # disability
}

TAIWAN_FEAT = {
    0: DataType.CONTINUOUS_REAL,  # limit bal
    1: DataType.CONTINUOUS_REAL,  # age
    2: DataType.CONTINUOUS_REAL,  # pay 0
    3: DataType.CONTINUOUS_REAL,  # pay 2
    4: DataType.CONTINUOUS_REAL,  # pay 3
    5: DataType.CONTINUOUS_REAL,  # pay 4
    6: DataType.CONTINUOUS_REAL,  # pay 5
    7: DataType.CONTINUOUS_REAL,  # pay 6
    8: DataType.CONTINUOUS_REAL,  # bill 1
    9: DataType.CONTINUOUS_REAL,  # bill 2
    10: DataType.CONTINUOUS_REAL,  # bill 3
    11: DataType.CONTINUOUS_REAL,  # bill 4
    12: DataType.CONTINUOUS_REAL,  # bill 5
    13: DataType.CONTINUOUS_REAL,  # bill 6
    14: DataType.CONTINUOUS_REAL,  # pay 1
    15: DataType.CONTINUOUS_REAL,  # pay 2
    16: DataType.CONTINUOUS_REAL,  # pay 3
    17: DataType.CONTINUOUS_REAL,  # pay 4
    18: DataType.CONTINUOUS_REAL,  # pay 5
    19: DataType.CONTINUOUS_REAL,  # pay 6
    20: DataType.DISCRETE,  # sex
    21: DataType.DISCRETE,  # education
    22: DataType.DISCRETE,  # marriage
}

WHO_FEAT = {
    0: DataType.CONTINUOUS_REAL,  # adult mortality
    1: DataType.CONTINUOUS_REAL,  # infant deaths
    2: DataType.CONTINUOUS_REAL,  # alcohol
    3: DataType.CONTINUOUS_REAL,  # percentage expenditure
    4: DataType.CONTINUOUS_REAL,  # hepatitis b
    5: DataType.CONTINUOUS_REAL,  # measles
    6: DataType.CONTINUOUS_REAL,  # bmi
    7: DataType.CONTINUOUS_REAL,  # under-five deaths
    8: DataType.CONTINUOUS_REAL,  # polio
    9: DataType.CONTINUOUS_REAL,  # total expenditure
    10: DataType.CONTINUOUS_REAL,  # diphtheria
    11: DataType.CONTINUOUS_REAL,  # hiv/aids
    12: DataType.CONTINUOUS_REAL,  # gdp
    13: DataType.CONTINUOUS_REAL,  # population
    14: DataType.CONTINUOUS_REAL,  # thinness 1-19 years
    15: DataType.CONTINUOUS_REAL,  # thinness 5-9 years
    16: DataType.CONTINUOUS_REAL,  # income composition of resources
    17: DataType.CONTINUOUS_REAL,  # schooling
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
                # gumbel_softmax sometimes gives NAN, let it try again if so. Fallback on regular softmax.
                tmp_x = F.gumbel_softmax(x[:, cat_idx: cat_end_idx], hard=hard)
                for i in range(2):
                    # gumbel_softmax sometimes returns nan, give it 2 changes to fix itself
                    if tmp_x.isnan().any():
                        tmp_x = F.gumbel_softmax(x[:, cat_idx: cat_end_idx], hard=hard)
                if tmp_x.isnan().any():
                    tmp_x = F.softmax(x[:, cat_idx: cat_end_idx], dim=-1)
                x[:, cat_idx: cat_end_idx] = tmp_x
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
        for col in self.cont_features:
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
        minmax_arr = []
        for i, name in enumerate(self.cont_features):
            min_val = min_vals[i]
            max_val = max_vals[i]
            df_copy[:, name] = (df_copy[:, name] - min_val) / (max_val - min_val)

            def minmax(x):
                # reverse the scaling we just did
                return x * (max_val - min_val) + min_val
            minmax_arr.append(minmax)
        
        return df_copy, minmax_arr

    def make_cont_features(self, data, cont_features):
        self.cont_features = [data.feat_var_map[i][0] for i in cont_features]
        self.min_vals = np.min(data.X[:, self.cont_features], axis=0)
        self.max_vals = np.max(data.X[:, self.cont_features], axis=0)


def load_data(data, label, feature_types, preprocessor=None):
    '''
         Load data and preprocess it. Adapted from Jiang et al. Uses a OHE for the data.
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
    data.num_features_processed = data.X.shape[1]
    if type(data.X) != type(torch.Tensor()):
        data.X = np.array(data.X)
    if need_make_cont_features:
        preprocessor.make_cont_features(data, cont_features)
    data.X, minmax = preprocessor.min_max_scale(data.X)

    return data, preprocessor, minmax


def prepare_data(args):
    ret = {"preprocessor": None, "train_data": None, "test_data": None, "model": None, "minmax": None}
    if args.finetune:
        modifier = 'shift'
    else:
        modifier = 'orig'
    if args.config["dataset_name"] == "german_credit":
        feature_types = CREDIT_FEAT
        train_data, preprocessor, minmax = load_data("data/german_train.csv", "credit_risk", feature_types)
        test_data, _, _ = load_data("data/german_test.csv", "credit_risk", feature_types, preprocessor)
        ret["preprocessor"] = preprocessor
        ret["minmax"] = minmax
    elif args.config["dataset_name"] == "heloc":
        feature_types = HELOC_FEAT
        train_data, preprocessor, minmax = load_data("data/heloc_train.csv", "label", feature_types)
        test_data, _, _ = load_data("data/heloc_test.csv", "label", feature_types, preprocessor)
        ret["preprocessor"] = preprocessor
        ret["minmax"] = minmax
    elif args.config["dataset_name"] == "ctg":
        feature_types = CTG_FEAT
        train_data, preprocessor,minmax = load_data("data/ctg_orig_train.csv", "label", feature_types)
        if args.finetune:
            train_data, _ , _ = load_data("data/ctg_shift_train.csv", "label", feature_types, preprocessor)
        test_data, _ , _ = load_data("data/ctg_" + modifier + "_test.csv", "label", feature_types, preprocessor)
        ret['preprocessor'] = preprocessor
        ret['minmax'] = minmax
    elif args.config["dataset_name"] == "student":
        feature_types = STUDENT_FEAT
        train_data, preprocessor, minmax = load_data("data/student_train.csv", "final_result", feature_types)
        test_data, _, _ = load_data("data/student_test.csv", "final_result", feature_types, preprocessor)
        ret['preprocessor'] = preprocessor
        ret['minmax'] = minmax
    elif args.config["dataset_name"] == "taiwan":
        feature_types = TAIWAN_FEAT
        train_data, preprocessor, minmax = load_data("data/taiwan_train.csv", "Y", feature_types)
        test_data, _, _ = load_data("data/taiwan_test.csv", "Y", feature_types, preprocessor)
        ret['preprocessor'] = preprocessor
        ret['minmax'] = minmax
    elif args.config["dataset_name"] == 'who':
        # will be who_orig or who_shift
        feature_types = WHO_FEAT
        train_data, preprocessor, minmax = load_data("data/who_" + modifier + "_train.csv", "label", feature_types)
        test_data, _,  _ = load_data("data/who_" + modifier + "_test.csv", "label", feature_types, preprocessor)
        ret['preprocessor'] = preprocessor
        ret['minmax'] = minmax
    else:
        raise NotImplementedError(f"Dataset {args.config['dataset_name']} not implemented")

    # reverse sort args.remove
    if args.remove_pct is not None:
        start_idx = args.remove_pct * 0.01 * len(train_data) * args.removal_start
        end_idx = args.remove_pct * 0.01 * len(train_data) * (args.removal_start + 1)
        train_data.X = np.concatenate((train_data.X[:int(start_idx)], train_data.X[int(end_idx):]), axis=0)
        train_data.y = np.concatenate((train_data.y[:int(start_idx)], train_data.y[int(end_idx):]), axis=0)

    ret["train_data"] = train_data
    ret["test_data"] = test_data
    return ret
