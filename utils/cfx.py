import time
import sys
import os
import copy

import pickle

import numpy as np
import torch
from tqdm import tqdm

from utils.dataset import DataType
from alibi.explainers import Counterfactual, cfproto
import tensorflow as tf
import utils.dataset

tf.compat.v1.disable_eager_execution()  # required for functionality like placeholder

# open a file with pickle
def open_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
# parts of this file copied from https://github.com/junqi-jiang/robust-ce-inn/blob/main/expnns/utilexp.py
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def get_clf_num_layers(model):
    if isinstance(model, torch.nn.Sequential):
        return model.num_hiddens
    return model.hidden_layer_sizes

def build_dataset_feature_types(columns, ordinal, discrete, continuous):
    feature_types = dict()
    for feat in ordinal.keys():
        feature_types[columns.index(feat)] = DataType.ORDINAL
    for feat in discrete.keys():
        feature_types[columns.index(feat)] = DataType.DISCRETE
    for feat in continuous:
        feature_types[columns.index(feat)] = DataType.CONTINUOUS_REAL
    return feature_types

class CFX_Generator:
    '''
        model - model
        dataset - dataset object containing X, y, feature_types
    '''

    def __init__(self, model, dataset, gap=0.1, desired_class=1, num_test_instances=50, num_layers=2):
        self.model = model
        self.num_layers = num_layers
        self.dataset = dataset
        self.X = dataset.X
        self.y = dataset.y
        # self.X2 = X2
        # self.y2 = y2
        # self.columns = columns
        # self.ordinal_features = ordinal_features
        # self.discrete_features = discrete_features
        # self.continuous_features = continuous_features
        # =1 (0) will select test instances with classification result 0 (1), =-1 will randomly select test instances
        self.desired_class = desired_class
        self.num_test_instances = num_test_instances

        # self.dataset = None
        # self.delta_min = -1
        # self.Mmax = None
        # self.delta_max = 0  # inf-d(clf, Mmax)
        # self.lof = None
        self.test_instances = None
        # self.inn_delta_non_0 = None
        # self.inn_delta_0 = None

        # load util
        # self.build_dataset_obj()
        # self.build_lof()
        # self.build_delta_min(gap)
        # self.build_Mplus_Mminus(gap)
        # self.build_Mmax()
        self.build_test_instances()
        # self.build_inns()

    def build_test_instances(self):
        ''' in the future, may want to, e.g., select test instances that have predicted class=0'''
        self.test_instances = self.X

    def setup_CE_arrays(self, CEs, is_CE, regenerate, num_to_run, num_features):
        if CEs is None:
            CEs = np.zeros((num_to_run, num_features))
        if is_CE is None:
            is_CE = np.zeros(num_to_run).astype(bool)
        if regenerate is None:
            regenerate = np.ones(num_to_run).astype(bool)
        return torch.tensor(CEs), is_CE, regenerate

    def setup_cat_vars(self, onehot):
        cat_var = {}
        i = 0
        if onehot:
            for idx in self.dataset.feature_types:
                if self.dataset.feature_types[idx] == DataType.DISCRETE:
                    num_vals = int(self.dataset.discrete_features[idx].item())
                    cat_var[i] = num_vals
                    i += num_vals
                elif self.dataset.feature_types[idx] == DataType.ORDINAL:
                    num_vals = int(self.dataset.ordinal_features[idx].item())
                    cat_var[i] = num_vals
                    i += num_vals
                elif self.dataset.feature_types[idx] == DataType.CONTINUOUS_REAL:
                    i += 1
        else:
            for idx in self.dataset.discrete_features.keys():
                num_vals = int(self.dataset.discrete_features[idx].item())
                cat_var[idx] = num_vals
            for idx in self.dataset.ordinal_features.keys():
                num_vals = int(self.dataset.ordinal_features[idx].item())
                cat_var[idx] = num_vals
        return cat_var

    def run_proto(self, kap=0, theta=10., scaler=None, test_instances=None, onehot=False, num_to_run = None,
                    CEs = None, is_CE = None, regenerate = None):
        if test_instances is None:
            test_instances = self.test_instances
        data_point = np.array(self.X[1])
        shape = (1,) + data_point.shape[:]
        predict_fn = lambda x: self.model(x)
        if num_to_run == None:
            num_to_run = len(test_instances)
        
        cat_var = self.setup_cat_vars(onehot)
        CEs, is_CE, regenerate = self.setup_CE_arrays(CEs, is_CE, regenerate, num_to_run, test_instances.shape[1])
        print("regenerating for indices ", np.where(regenerate)[0])

        start_time = time.time()

        rng = (0., 1.)  # scale features between 0 and 1
        rng_shape = (1, len(self.dataset.feature_types)) # needs to be defined as original (not OHE) feature space
        feature_range = ((np.ones(rng_shape) * rng[0]).astype(np.float32), 
                 (np.ones(rng_shape) * rng[1]).astype(np.float32))
        
        if cat_var == {}:
            # only continuous features
            cf = cfproto.CounterfactualProto(predict_fn, shape, use_kdtree=True, theta=theta, kappa=kap,
                                             feature_range=feature_range)
            cf.fit(self.X, trustscore_kwargs=None)
        else:
            cf = cfproto.CounterfactualProto(predict_fn, shape, use_kdtree=True, theta=theta, feature_range = feature_range,
                                             cat_vars=cat_var, kappa=kap, ohe=onehot, beta = 0.01, c_init = 1.0, c_steps = 5,
                                             max_iterations=500, eps=(1e-2, 1e-2), update_num_grad=1)
            cf.fit(self.X)
        for i, x in tqdm(enumerate(self.test_instances)):
            if i == num_to_run:
                break
            if not regenerate[i]:
                print("skipping index ",i)
                continue
            this_point = x
            with HiddenPrints():
                explanation = cf.explain(this_point.reshape(1, -1), Y=None, target_class=None, k=20, k_type='mean',
                                         threshold=0., verbose=True, print_every=10, log_every=10)
            if explanation is None or explanation["cf"] is None:
                CEs[i,:] = torch.tensor(this_point)
                is_CE[i] = 0
                continue
            this_cf = explanation["cf"]["X"]
            this_cf = np.array(this_cf[0])
            CEs[i,:] = torch.tensor(this_cf)
            is_CE[i] = 1
        print("total computation time in s:", time.time() - start_time)

        # save CEs to file
        with open('CEsProto.pkl', 'wb') as f:
            if scaler is not None:
                pickle.dump(scaler.inverse_transform(CEs), f)
            else:
                pickle.dump(CEs, f)
        with open('CEsProtoBool.pkl', 'wb') as f:
            pickle.dump(is_CE, f)

        return torch.tensor(np.array(CEs).astype('float32')).squeeze(),\
                     torch.tensor(np.array(is_CE).astype('bool')).squeeze()

    def run_wachter(self, lam_init=0.0001, max_iter=100, max_lam_steps=10, target_proba=0.6, scaler=None,
                    test_instances=None, num_to_run = None, CEs = None, is_CE = None, regenerate = None):
        ''' 
        This algorithm isn't great -- no logical constraints on feature values so finds non-integral
        values for categorical features. As far as I can tell, no way to change this.

        CEs - list of counterfactuals, can be passed in to save previous CFX if we only want to update certain
                instances
        is_CE - list of booleans of whether each CFX generation was successful
        '''

        if test_instances is None:
            test_instances = self.test_instances
        if num_to_run is None:
            num_to_run = len(test_instances)

        CEs, is_CE, regenerate = self.setup_CE_arrays(CEs, is_CE, regenerate, num_to_run, test_instances.shape[1])
        print("regenerating for indices ", np.where(regenerate)[0])

        data_point = np.array(self.X[1])
        shape = (1,) + data_point.shape[:]
        predict_fn = lambda x: self.model(x)

        eps = 0.1  # was 0.01
        tol = 0.1  # was 0.05
        cf = Counterfactual(predict_fn, shape, distance_fn='l1', target_proba=target_proba,
                            target_class='other', max_iter=max_iter, early_stop=50, lam_init=lam_init,
                            max_lam_steps=max_lam_steps, tol=tol, learning_rate_init=0.1,
                            feature_range=(0, 1), eps=eps, init='identity',
                            decay=True, write_dir=None, debug=False)
        start_time = time.time()
        for i, x in enumerate(tqdm(test_instances)):
            if i == num_to_run:
                break
            if not regenerate[i]:
                print("skipping index ",i)
                continue
            this_point = x
            with HiddenPrints():
                explanation = cf.explain(this_point.reshape(1, -1))
            if explanation is None or explanation['cf'] is None:
                CEs[i,:] = torch.tensor(this_point)
                is_CE[i] = 0
                continue
            this_cf = explanation["cf"]["X"]
            CEs[i,:] = torch.tensor(np.array(this_cf[0]))
            is_CE[i] = 1
        print("total computation time in s:", time.time() - start_time)

        # save CEs to file
        with open('CEsWachter.pkl', 'wb') as f:
            if scaler is not None:
                pickle.dump(scaler.inverse_transform(CEs), f)
            else:
                pickle.dump(CEs, f)
        with open('CEsWachterBool.pkl', 'wb') as f:
            pickle.dump(is_CE, f)

        return torch.tensor(np.array(CEs).astype('float32')).squeeze(), \
                    torch.tensor(np.array(is_CE).astype('bool')).squeeze()
