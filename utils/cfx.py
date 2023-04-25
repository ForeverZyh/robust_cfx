import time
import sys
import os
import copy

import numpy as np
from tqdm import tqdm

from alibi.explainers import Counterfactual,


# parts of this file copied from https://github.com/junqi-jiang/robust-ce-inn/blob/main/expnns/utilexp.py
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def get_clf_num_layers(clf):
    if isinstance(clf.hidden_layer_sizes, int):
        return 3
    else:
        return len(clf.hidden_layer_sizes) + 2

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
    def __init__(self, clf, X1, y1, X2, y2, columns, ordinal_features, discrete_features, continuous_features,
                 feature_var_map, gap=0.1, desired_class=1, num_test_instances=50):
        self.clf = clf
        self.num_layers = get_clf_num_layers(clf)
        self.X1 = X1
        self.y1 = y1
        self.X2 = X2
        self.y2 = y2
        self.columns = columns
        self.ordinal_features = ordinal_features
        self.discrete_features = discrete_features
        self.continuous_features = continuous_features
        self.feat_var_map = feature_var_map
        # =1 (0) will select test instances with classification result 0 (1), =-1 will randomly select test instances
        self.desired_class = desired_class
        self.num_test_instances = num_test_instances

        self.dataset = None
        self.delta_min = -1
        self.Mmax = None
        self.delta_max = 0  # inf-d(clf, Mmax)
        self.lof = None
        self.test_instances = None
        self.inn_delta_non_0 = None
        self.inn_delta_0 = None

        # load util
        self.build_dataset_obj()
        self.build_lof()
        self.build_delta_min(gap)
        #self.build_Mplus_Mminus(gap)
        self.build_Mmax()
        self.build_test_instances()
        self.build_inns()

    def build_dataset_obj(self):
        self.dataset = Dataset(len(self.columns) - 1, self.clf.n_features_in_,
                               build_dataset_feature_types(self.columns, self.ordinal_features, self.discrete_features,
                                                           self.continuous_features), self.feat_var_map)

    def build_lof(self):
        self.lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
        self.lof.fit(self.X1.values)

    def build_delta_min(self, gap):
        wb_orig = get_flattened_weight_and_bias(self.clf)
        for i in range(5):
            np.random.seed(i)
            idxs = np.random.choice(range(len(self.X2.values)), int(gap * len(self.X2.values)))
            this_clf = copy.deepcopy(self.clf)
            this_clf.partial_fit(self.X2.values[idxs], self.y2.values[idxs])
            this_wb = get_flattened_weight_and_bias(this_clf)
            this_delta = inf_norm(wb_orig, this_wb)
            if this_delta >= self.delta_min:
                self.delta_min = this_delta

    def build_Mplus_Mminus(self, gap):
        self.build_delta_min(gap)
        self.Mplus, self.Mminus = build_delta_extreme_shifted_models(self.clf, self.delta_min)

    def build_Mmax(self):
        wb_orig = get_flattened_weight_and_bias(self.clf)
        for i in range(5):
            np.random.seed(i)
            idxs = np.random.choice(range(len(self.X2.values)), int(0.99 * len(self.X2.values)))
            this_clf = copy.deepcopy(self.clf)
            this_clf.partial_fit(self.X2.values[idxs], self.y2.values[idxs])
            this_wb = get_flattened_weight_and_bias(this_clf)
            this_delta = inf_norm(wb_orig, this_wb)
            if this_delta >= self.delta_max:
                self.delta_max = this_delta
                self.Mmax = this_clf

        wb_orig = get_flattened_weight_and_bias(self.clf)
        wb_max = get_flattened_weight_and_bias(self.Mmax)
        self.delta_max = inf_norm(wb_max, wb_orig)

    def build_test_instances(self):
        np.random.seed(1)
        if self.desired_class >= 0:
            if self.desired_class == 1:
                random_idx = np.where(self.clf.predict(self.X1.values) == 0)[0]
            else:
                random_idx = np.where(self.clf.predict(self.X1.values) == 1)[0]
            random_idx = np.random.choice(random_idx, min(self.num_test_instances, len(random_idx)))
        else:
            random_idx = np.random.randint(len(self.X1.values) - 1, size=(self.num_test_instances,))
        self.test_instances = self.X1.values[random_idx]

    def build_inns(self):
        delta = self.delta_min
        nodes = build_inn_nodes(self.clf, self.num_layers)
        weights, biases = build_inn_weights_biases(self.clf, self.num_layers, delta, nodes)
        self.inn_delta_non_0 = Inn(self.num_layers, delta, nodes, weights, biases)
        delta = 0
        weights_0, biases_0 = build_inn_weights_biases(self.clf, self.num_layers, delta, nodes)
        self.inn_delta_0 = Inn(self.num_layers, delta, nodes, weights_0, biases_0)

    def run_wachter(self, lam_init=0.1, max_lam_steps=10, target_proba=0.6):
        CEs = []
        data_point = np.array(self.X1.values[1])
        shape = (1,) + data_point.shape[:]
        predict_fn = lambda x: self.clf.predict_proba(x)
        cf = Counterfactual(predict_fn, shape, distance_fn='l1', target_proba=target_proba,
                            target_class='other', max_iter=1000, early_stop=50, lam_init=lam_init,
                            max_lam_steps=max_lam_steps, tol=0.05, learning_rate_init=0.1,
                            feature_range=(0, 1), eps=0.01, init='identity',
                            decay=True, write_dir=None, debug=False)
        start_time = time.time()
        for i, x in tqdm(enumerate(self.test_instances)):
            this_point = x
            with HiddenPrints():
                explanation = cf.explain(this_point.reshape(1, -1))
            if explanation is None:
                CEs.append(None)
                continue
            if explanation["cf"] is None:
                CEs.append(None)
                continue
            proto_cf = explanation["cf"]["X"]
            proto_cf = proto_cf[0]
            this_cf = np.array(proto_cf)
            CEs.append(this_cf)
        print("total computation time in s:", time.time() - start_time)
        assert len(CEs) == len(self.test_instances)
        return CEs