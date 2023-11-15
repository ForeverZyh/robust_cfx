import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import LocalOutlierFactor
from inn import *
from utils.dataset import DataType, Dataset
import os, sys
from tqdm import tqdm
from roar.recourse_methods import RobustRecourse
from roar.recourse_utils import lime_explanation
from alibi.explainers import Counterfactual, cfproto
import tensorflow as tf
import time
from utils.optsolver import OptSolver

tf.get_logger().setLevel(40)
tf.compat.v1.disable_v2_behavior()



def get_flattened_weight_and_bias(clf, weights=True, biases=True):
    if not weights and not biases:
        return 0
    w_concat = []
    b_concat = []
    if weights:
        w_all = clf.coefs_
        w_concat = np.append(w_all[0].flatten(), w_all[1].flatten())
    if biases:
        b_all = clf.intercepts_
        b_concat = np.append(b_all[0].flatten(), b_all[1].flatten())
    wb_concat = np.append(w_concat, b_concat)
    return wb_concat


def inf_norm(x, y):
    return np.max(abs(x - y))


def inf_norm_percentage(x, y):
    return np.max(abs(x - y) / abs(x))


def calculate_avg_2_dist(hidden_layer_size_val, X_train, y_train):
    training_sizes = [int(i * len(X_train)) for i in np.arange(0.5, 1.01, 0.05)]
    clfs = []
    wbs = []  # flattened concatenated weights and biases
    for size in training_sizes:
        this_clf = MLPClassifier(learning_rate='adaptive', hidden_layer_sizes=hidden_layer_size_val,
                                 learning_rate_init=0.01, batch_size=32, max_iter=5000, random_state=11)
        this_clf.fit(X_train[:size], y_train[:size])
        clfs.append(this_clf)
        wbs.append(get_flattened_weight_and_bias(this_clf))

    # calculate p=2-distance
    dists_wbs = []
    dists_wbs_norm = []
    for i in range(1, len(wbs)):
        dists_wbs.append(round(euclidean_distances(wbs[i - 1].reshape(1, -1), wbs[i].reshape(1, -1)).flatten()[0], 3))
        dists_wbs_norm.append(round(euclidean_distances((wbs[i - 1] / wbs[0].shape[0]).reshape(1, -1),
                                                        (wbs[i] / wbs[0].shape[0]).reshape(1, -1)).flatten()[0], 3))

    avg_dist = np.mean(dists_wbs)
    avg_dist_norm = np.mean(dists_wbs_norm)
    return avg_dist, avg_dist_norm


def incremental_train(train_gap, base_clf, X_train, y_train, ignore_first_model=False, percentage_normalise=True):
    """
    function for incremental training on D2
    :param train_gap: increment when training on D2. 0.05 means train 5%, 10%, ..., 95%, 100% on D2
    :param base_clf: NN trained on D1
    :param X_train: training set of D2
    :param y_train: same
    :param ignore_first_model: ignore the change between the base model and the first incremented model
    :param percentage_normalise: if True, calculate percentage change of weights instead of absolute change
    :return: inf-distance of weights and biases
    """
    # training_sizes = [int(i * len(X_train)) for i in (np.arange(0, 1.01, train_gap))[1:]]
    training_sizes = [int(i * len(X_train)) for i in (np.arange(0, 1.01, train_gap))]
    ws = []  # flattened weights for every classifier in the range
    bs = []

    this_clf = copy.copy(base_clf)
    ws.append(get_flattened_weight_and_bias(this_clf, True, False))
    bs.append(get_flattened_weight_and_bias(this_clf, False, True))

    # incremental training
    for i, size in enumerate(training_sizes):
        if i == 0:
            continue
        inc_idx_l = training_sizes[i - 1] + 1
        inc_idx_r = training_sizes[i]
        this_clf = this_clf.partial_fit(X_train[inc_idx_l:inc_idx_r], y_train[inc_idx_l:inc_idx_r], [0, 1])
        ws.append(get_flattened_weight_and_bias(this_clf, True, False))
        bs.append(get_flattened_weight_and_bias(this_clf, False, True))

    # calculate inf-distance
    dists_ws = []
    dists_bs = []

    for i in range(1, len(ws)):
        if ignore_first_model and i == 1:
            continue
        if not percentage_normalise:
            # dists_ws.append(round(inf_norm(ws[i - 1], ws[i]), 3))
            dists_ws.append(round(inf_norm(ws[0], ws[i]), 3))  # calculate differences with the first model
            # dists_bs.append(round(inf_norm(bs[i - 1], bs[i]), 3))
            dists_bs.append(round(inf_norm(bs[0], bs[i]), 3))
        else:
            dists_ws.append(round(inf_norm_percentage(ws[i - 1], ws[i]), 3))
            dists_bs.append(round(inf_norm_percentage(bs[i - 1], bs[i]), 3))
    avg_dist_w = np.mean(dists_ws)
    avg_dist_b = np.mean(dists_bs)

    return dists_ws, avg_dist_w, ws, dists_bs, avg_dist_b, bs


def plot_incremental_training_weight_bias_changes(training_gap, hidden_size, lr, batch_sz, max_iter, d1_x, d1_y, d2_x,
                                                  d2_y, percentage_normalise=False):
    avg_dists_w = []
    avg_dists_b = []
    h_ds_size = (np.arange(0, 1.01, training_gap))[1:]
    figw, axsw = plt.subplots(3, 3)
    figw.tight_layout(rect=[0, 0.1, 1, 0.95])
    figb, axsb = plt.subplots(3, 3)
    figb.tight_layout(rect=[0, 0.03, 1, 0.95])
    print(
        f"inf-distance for weights and biases of classifiers with hidden size {hidden_size}, training gap is {training_gap}, ")
    print("WEIGHTS or BIASES inf-distance /// % of D2 used")
    for i, h_size in enumerate(hidden_size):
        # for this hidden layer size, train a base clf
        this_clf = MLPClassifier(learning_rate='adaptive', hidden_layer_sizes=h_size, learning_rate_init=lr,
                                 batch_size=batch_sz,
                                 max_iter=max_iter, random_state=0)
        this_clf.fit(d1_x, d1_y)

        this_dists_w, this_avg_w, _, this_dists_b, this_avg_b, _ = incremental_train(training_gap, this_clf, d2_x, d2_y,
                                                                                     percentage_normalise=percentage_normalise)
        avg_dists_w.append(this_avg_w)
        avg_dists_b.append(this_avg_b)
        axsw[int(i / 3), i % 3].plot(h_ds_size, this_dists_w)
        axsw[int(i / 3), i % 3].set_title(f'#{h_size}, weights')
        axsb[int(i / 3), i % 3].plot(h_ds_size, this_dists_b)
        axsb[int(i / 3), i % 3].set_title(f'#{h_size}, biases')

    plt.figure(3)
    plt.plot(hidden_size, avg_dists_w, 'ro-')
    plt.title('avg WEIGHTS inf-distance /// hidden layer sizes')

    plt.figure(4)
    plt.plot(hidden_size, avg_dists_b, 'ro-')
    plt.title('avg BIASES inf-distance /// hidden layer sizes')
    plt.show()
    return avg_dists_w, avg_dists_b


def get_clf_num_layers(clf):
    if isinstance(clf.hidden_layer_sizes, int):
        return 3
    else:
        return len(clf.hidden_layer_sizes) + 2


def build_inn_nodes(clf, num_layers):
    nodes = dict()
    for i in range(num_layers):
        this_layer_nodes = []
        if i == 0:
            num_nodes_i = clf.n_features_in_
        elif i == num_layers - 1:
            num_nodes_i = 1
        else:
            if isinstance(clf.hidden_layer_sizes, int):
                num_nodes_i = clf.hidden_layer_sizes
            else:
                num_nodes_i = clf.hidden_layer_sizes[i - 1]

        for j in range(num_nodes_i):
            this_layer_nodes.append(Node(i, j))
        nodes[i] = this_layer_nodes
    return nodes


def build_inn_weights_biases(clf, num_layers, delta, nodes):
    ws = clf.coefs_
    bs = clf.intercepts_
    weights = dict()
    biases = dict()
    for i in range(num_layers - 1):
        for node_from in nodes[i]:
            for node_to in nodes[i + 1]:
                # round by 4 decimals
                w_val = round(ws[i][node_from.index][node_to.index], 8)
                weights[(node_from, node_to)] = Interval(w_val, w_val - delta, w_val + delta)
                b_val = round(bs[i][node_to.index], 8)
                biases[node_to] = Interval(b_val, b_val - delta, b_val + delta)
    return weights, biases


def build_dataset_feature_types(columns, ordinal, discrete, continuous):
    feature_types = dict()
    for feat in ordinal.keys():
        feature_types[columns.index(feat)] = DataType.ORDINAL
    for feat in discrete.keys():
        feature_types[columns.index(feat)] = DataType.DISCRETE
    for feat in continuous:
        feature_types[columns.index(feat)] = DataType.CONTINUOUS_REAL
    return feature_types


def normalised_l1(xp, x, dataset):
    """
    function to calculate normalised L1 between two flat arrays
    :param xp: flat np array of shape (num_vars,)
    :param x: flat np array of shape (num_vars,)
    :param dataset: instance of class Dataset
    :return: float number
    """
    dist = 0
    for feat_idx in range(dataset.num_features):
        var_idx = dataset.feat_var_map[feat_idx]
        # discrete: not the same -> 1
        if dataset.feature_types[feat_idx] == DataType.DISCRETE:
            dist += np.max(np.abs(xp[var_idx] - x[var_idx]))
        # ordinal: |sum(xpi) - sum(xi)| / (k-1), where k is number of possible values
        if dataset.feature_types[feat_idx] == DataType.ORDINAL:
            dist += np.abs(np.sum(xp[var_idx]) - np.sum(x[var_idx])) / (len(var_idx) - 1)
        # continuous: |xpi - xi|/(ubi-lbi) <==> |xpi - xi|, because (ub-lb)=1
        if dataset.feature_types[feat_idx] == DataType.CONTINUOUS_REAL:
            dist += np.abs(np.sum(xp[var_idx]) - np.sum(x[var_idx]))
    return round(dist / dataset.num_features, 3)


def normalised_l0(xp, x, dataset):
    dist = 0
    for feat_idx in range(dataset.num_features):
        var_idx = dataset.feat_var_map[feat_idx]
        if dataset.feature_types[feat_idx] == DataType.DISCRETE:
            if np.max(np.abs(xp[var_idx] - x[var_idx])) >= 0.0001:
                dist += 1
        else:
            if np.abs(np.sum(xp[var_idx]) - np.sum(x[var_idx])) >= 0.0001:
                dist += 1
    return round(dist / dataset.num_features, 3)


def normalised_l1_all(xp, x):
    """
    function to calculate normalised l1 distance between two flat arrays,
    does not distinguish between different types of features
    :param xp: flat np array of shape (num_vars,)
    :param x: flat np array of shape (num_vars,)
    :return: float number
    """
    return np.sum(np.abs(xp - x)) / (xp.shape[0])


def build_delta_extreme_shifted_models(clf, delta):
    """
    function to build the 2 most shifted models under delta-plausible model shifts
    :param clf: original model
    :param delta: delta
    :return: clf_plus, clf_minus
    """
    clf_plus = copy.deepcopy(clf)
    for i in range(len(clf_plus.coefs_)):
        clf_plus.coefs_[i] += delta
    for i in range(len(clf_plus.intercepts_)):
        clf_plus.intercepts_[i] += delta

    clf_minus = copy.deepcopy(clf)
    for i in range(len(clf_minus.coefs_)):
        clf_minus.coefs_[i] += delta
    for i in range(len(clf_minus.intercepts_)):
        clf_minus.intercepts_[i] += delta

    return clf_plus, clf_minus


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class UtilExp:
    def __init__(self, clf, preprocessor, train_X, test_instances, test_labels, dataset = None, target_p=0):
        self.clf = clf
        self.preprocessor = preprocessor
        # self.num_layers = get_clf_num_layers(clf)
        # self.X1 = X1
        # self.y1 = y1
        # self.X2 = X2
        # self.y2 = y2
        # self.columns = columns
        # self.ordinal_features = ordinal_features
        # self.discrete_features = discrete_features
        # self.continuous_features = continuous_features
        # self.feat_var_map = feature_var_map
        # # =1 (0) will select test instances with classification result 0 (1), =-1 will randomly select test instances
        # self.desired_class = desired_class
        # self.num_test_instances = num_test_instances

        self.dataset = dataset
        self.delta_min = -1
        self.Mmax = None
        self.delta_max = 0  # inf-d(clf, Mmax)
        self.lof = None
        self.train_X = train_X
        self.test_instances = test_instances
        self.test_labels = test_labels
        self.inn_delta_non_0 = None
        self.inn_delta_0 = None
        self.target_p = target_p

        # # load util
        # self.build_dataset_obj()
        # self.build_lof()
        # self.build_delta_min(gap)
        # #self.build_Mplus_Mminus(gap)
        # self.build_Mmax()
        # self.build_test_instances()
        # self.build_inns()

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

    def verify_soundness(self, update_test_instances=False):
        sound = 0
        valids = []
        valid_instances = []
        for i, x in enumerate(self.test_instances):
            y_prime = self.clf.predict(x.reshape(1, -1))[0]
            this_solver = OptSolver(self.dataset, self.inn_delta_non_0, y_prime, x, mode=1, M=10000, x_prime=x)
            found, bound = this_solver.compute_inn_bounds()
            if found == 1:
                sound += 1
                valids.append(i)
        print(f"percentage of sound model changes: {sound / len(self.test_instances)}")
        if update_test_instances:
            if len(valids) >= 50:
                valids = valids[:50]
            self.test_instances = self.test_instances[valids]
            print(f"test instances updated to sound (x, Delta) pairs, length: {len(valids)}")
        return valids

    def is_robust_raw(self, x, cf):
        y_prime = 1 if self.clf.predict(x.reshape(1, -1))[0] == 0 else 0
        this_solver = OptSolver(self.dataset, self.inn_delta_non_0, y_prime, x, mode=1, M=10000, x_prime=cf)
        found, bound = this_solver.compute_inn_bounds()
        return found, bound

    def is_robust(self, x, cf):
        """
        run robustness test for delta=delta_min:
        y'=1, lb(INN) >= 0 <---> robust
        y'=0, ub(INN) <= 0 <---> robust
        :param x: x, (n features,)
        :param cf: cf, (n features,)
        :return: boolean
        """
        y_prime = 1 if self.clf.predict(x.reshape(1, -1))[0] == 0 else 0
        this_solver = OptSolver(self.dataset, self.inn_delta_non_0, y_prime, x, mode=1, M=10000, x_prime=cf)
        found, bound = this_solver.compute_inn_bounds()
        if found == 1:
            return True
        else:
            return False

    def is_robust_custom_delta(self, x, cf, delta):
        nodes = build_inn_nodes(self.clf, self.num_layers)
        weights, biases = build_inn_weights_biases(self.clf, self.num_layers, delta, nodes)
        inn = Inn(self.num_layers, delta, nodes, weights, biases)
        y_prime = 1 if self.clf.predict(x.reshape(1, -1))[0] == 0 else 0
        this_solver = OptSolver(self.dataset, inn, y_prime, x, mode=1, M=10000, x_prime=cf)
        found, bound = this_solver.compute_inn_bounds()
        if found == 1:
            return True
        else:
            return False

    def evaluate_one(self, this_cf, x):
        # function for calculating evaluation metrics for one **non-null** counterfactual instance
        found_valid, cf_valid, delta_valid, m2_valid, l1s, l0s, lofs = 0, 0, 0, 0, 0, 0, 0
        if this_cf is not None:
            found_valid += 1
            if self.clf.predict(this_cf.reshape(1, -1))[0] != self.clf.predict(x.reshape(1, -1))[0]:
                cf_valid += 1
                l1s += normalised_l1_all(this_cf, x)
                l0s += normalised_l0(this_cf, x, self.dataset)
                lofs += self.lof.predict(this_cf.reshape(1, -1))[0]
            if self.is_robust(x, this_cf):
                delta_valid += 1
            if self.clf.predict(this_cf.reshape(1, -1))[0] != self.clf.predict(x.reshape(1, -1))[0] and \
                    self.Mmax.predict(
                        this_cf.reshape(1, -1))[0] != self.clf.predict(x.reshape(1, -1))[0]:
                m2_valid += 1
        return found_valid, cf_valid, delta_valid, m2_valid, l1s, l0s, lofs

    def evaluate_ces(self, cf):
        found_valids, cf_valids, delta_valids, m2_valids, l1ss, l0ss, lofss = 0, 0, 0, 0, 0, 0, 0
        for i, this_cf in enumerate(cf):
            x = self.test_instances[i]
            if this_cf is not None:
                found_valid, cf_valid, delta_valid, m2_valid, l1s, l0s, lofs = self.evaluate_one(this_cf, x)
                found_valids += found_valid
                cf_valids += cf_valid
                delta_valids += delta_valid
                m2_valids += m2_valid
                l1ss += l1s
                l0ss += l0s
                lofss += lofs
        print("found:", found_valids / len(self.test_instances))
        if cf_valids != 0:
            print("average normalised L1:", l1ss / cf_valids)
            print("average normalised L0:", l0ss / cf_valids)
            print("average lof score:", lofss / cf_valids)
        print("counterfactual validity:", cf_valids / len(self.test_instances))
        print("delta validity:", delta_valids / len(self.test_instances))
        print("m2 validity:", m2_valids / len(self.test_instances))

    def run_INN_custom_delta(self, delta):
        return self.run_INN(robust=True, delta=delta)

    def run_INN_max_robust(self):
        delta = round(self.delta_max * 1.02, 4)
        return self.run_INN(robust=True, delta=delta)

    def run_INN_robust(self):
        return self.run_INN(robust=True, delta=None)

    def run_INN_non_robust(self):
        return self.run_INN(robust=False, delta=None)

    def run_INN(self, robust=False, delta=None):
        start_time = time.time()
        CEs = []
        is_cfx = []
        for i, (x,y) in tqdm(enumerate(zip(self.test_instances, self.test_labels))):
            if robust is False:
                this_cf = self.run_INN_one(x, y)
            else:
                this_cf = self.run_INN_one_delta_robust(x, delta=delta)
            CEs.append(this_cf)
            print("\ndiff", (x-this_cf).sum())
            print("y: ",y)
            print("pred is ",self.clf.predict(this_cf.reshape(1,-1)))
            is_cfx.append(self.clf.predict(this_cf.reshape(1, -1))[0] != y)
        print("total computation time in s:", time.time() - start_time)
        assert len(CEs) == len(self.test_instances)
        return CEs, is_cfx

    def run_INN_one(self, x, y):
        y_prime = 1 if y == 0 else 0
        # changing inn_delta_0 to inn_delta-non_0 here and in next fx
        this_solver = OptSolver(self.dataset, self.inn_delta_non_0, y_prime, x, mode=0, eps=0.01, x_prime=None)
        this_cf = this_solver.compute_counterfactual()
        if this_cf is None:
            return this_cf
        else:
            return np.round(this_cf, 5)

    def run_INN_one_delta_robust(self, x, delta=None):
        y_prime = 1 if self.clf.predict(x.reshape(1, -1))[0] == 0 else 0
        eps = 0.01
        this_solver = OptSolver(self.dataset, self.inn_delta_non_0, y_prime, x, mode=0, eps=0.01, x_prime=None)
        this_cf = this_solver.compute_counterfactual()
        count = 0
        if delta is None:  # default, delta=self.delta_min
            while self.is_robust(x, this_cf) != 1 and this_cf is not None and eps <= 20:
                this_solver = OptSolver(self.dataset, self.inn_delta_non_0, y_prime, x, mode=0, eps=eps, x_prime=None)
                this_cf = this_solver.compute_counterfactual()
                eps += 0.2
                count += 1
        else:
            while self.is_robust_custom_delta(x, this_cf, delta=delta) != 1 and this_cf is not None:
                this_solver = OptSolver(self.dataset, self.inn_delta_non_0, y_prime, x, mode=0, eps=eps, x_prime=None)
                this_cf = this_solver.compute_counterfactual()
                eps += 0.2
        return this_cf

    def run_ROAR(self, robust=False, labels=(1,), delta=None, lamb1_mul=4, max_iter=10, eps=1):
        CEs = []
        is_cfx = []
        # find categorical features
        cat_feats = []
        cont_feats = set(self.preprocessor.cont_features)
        for v in self.preprocessor.feature_var_map:
            if v not in cont_feats:
                cat_feats.extend(self.preprocessor.feature_var_map[v])
        cat_feats.sort()
        start_time = time.time()
        valid = 0
        cnt = 0
        with tqdm(total=len(self.test_instances)) as pbar:
            for i, (x, label) in enumerate(zip(self.test_instances, labels)):
                lamb1 = 1
                if not robust:
                    ce, _ = self.run_roar_one(x, cat_feats, (0,), lamb1=1, lamb22=None, eps=1 - label)
                else:
                    ce = self.run_roar_one_delta_robust(x, cat_feats, (0,), lamb1=lamb1, delta=None,
                                                        lamb1_mul=lamb1_mul, max_iter=max_iter)
                if ce is None:
                    ce = x
                CEs.append(ce)
                is_cfx.append(self.clf.predict(ce.reshape(1, -1))[0] != label)
                # print(label, is_cfx[-1])
                pbar.update(1)
                cnt += 1
                valid += is_cfx[-1]
                pbar.set_postfix({'valid': round(valid / cnt * 100, 2)})
        print("total computation time in s:", time.time() - start_time)
        assert len(CEs) == len(self.test_instances)
        return CEs, is_cfx

    def roar_util(self, recourses, xs):
        assert len(recourses) == len(xs)
        delta_valid = 0
        for i, r in enumerate(recourses):
            if self.is_robust(xs[i], r):
                delta_valid += 1
        return delta_valid / len(xs)

    def run_roar_one(self, x, cat_feats, labels, lamb1=1, lamb22=None, eps=1):

        coefficients = intercept = None
        robust_recourse = RobustRecourse(W=coefficients, W0=intercept, feature_costs=None, y_target=eps, target_p=self.target_p)
        with HiddenPrints():
            if lamb22 is None:
                lamb2 = robust_recourse.choose_lambda(x.reshape(1, -1), self.clf.predict, self.train_X,
                                                      self.clf.predict_logits, cat_feats=cat_feats)
            else:
                lamb2 = lamb22
        coefficients, intercept = lime_explanation(self.clf.predict_logits, self.train_X, x, cat_feats=cat_feats)
        coefficients, intercept = np.round_(coefficients, 4), np.round_(intercept, 4)
        robust_recourse.set_W(coefficients)
        robust_recourse.set_W0(intercept)
        r, delta_r = robust_recourse.get_recourse(x, lamb1=lamb1, lamb2=lamb2)
        if len(cat_feats) > 0:
            r[cat_feats] = x[cat_feats]
        if r is not None:
            return r, lamb2
        else:
            return None, None

    def run_roar_one_delta_robust(self, x, cat_feats, labels, lamb1=1, delta=None, lamb1_mul=4, max_iter=10):
        # find the hyperparameter that produces the best bound, if not found robust
        count = 0
        best_cf, lamb2 = self.run_roar_one(x, cat_feats, labels, lamb1)
        found, bound = self.is_robust_raw(x, best_cf)
        if found == 1:
            return best_cf
        best_bound = -100000 if bound is None else bound
        if delta is None:  # default, delta=self.delta_min
            while count <= max_iter:
                this_cf, lamb2 = self.run_roar_one(x, cat_feats, labels, lamb1, lamb2)
                found, bound = self.is_robust_raw(x, this_cf)
                if found == 1:
                    return this_cf
                if bound is None:
                    lamb1 = lamb1 * lamb1_mul
                    lamb2 = lamb2 / 2
                    count += 1
                    continue
                if bound >= best_bound:
                    best_bound = bound
                    best_cf = this_cf
                lamb1 = lamb1 * lamb1_mul
                lamb2 = lamb2 / 2
                count += 1
        else:
            raise NotImplementedError("custom delta robust for ROAR not implemented")

        return best_cf

    def run_proto(self, kap=0.1, theta=0.):
        data_point = np.array(self.X1.values[1])
        shape = (1,) + data_point.shape[:]
        predict_fn = lambda x: self.clf.predict_proba(x)
        cat_var = {}
        for idx in self.dataset.feature_types:
            if self.dataset.feature_types[idx] != DataType.CONTINUOUS_REAL:
                for varidx in self.dataset.feat_var_map[idx]:
                    cat_var[varidx] = 2
        CEs = []
        start_time = time.time()
        if len(self.discrete_features.keys()) == 0 and len(self.ordinal_features.keys()) == 0:
            cf = cfproto.CounterFactualProto(predict_fn, shape, use_kdtree=True, theta=theta, kappa=kap,
                                             feature_range=(np.array(self.X1.values.min(axis=0)).reshape(1, -1),
                                                            np.array(self.X1.values.max(axis=0)).reshape(1, -1)))
            cf.fit(self.X1.values, trustscore_kwargs=None)
        else:
            cf = cfproto.CounterFactualProto(predict_fn, shape, use_kdtree=True, theta=theta, feature_range=(
                np.array(self.X1.min(axis=0)).reshape(1, -1), np.array(self.X1.max(axis=0)).reshape(1, -1)),
                                             cat_vars=cat_var, kappa=kap,
                                             ohe=False)
            cf.fit(self.X1.values)
        for i, x in tqdm(enumerate(self.test_instances)):
            this_point = x
            with HiddenPrints():
                explanation = cf.explain(this_point.reshape(1, -1), Y=None, target_class=None, k=20, k_type='mean',
                                         threshold=0., verbose=True, print_every=100, log_every=100)
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

    def run_proto_robust(self, kap=0.1, theta=0.):
        data_point = np.array(self.X1.values[1])
        shape = (1,) + data_point.shape[:]
        predict_fn = lambda x: self.clf.predict_proba(x)
        cat_var = {}
        for idx in self.dataset.feature_types:
            if self.dataset.feature_types[idx] != DataType.CONTINUOUS_REAL:
                for varidx in self.dataset.feat_var_map[idx]:
                    cat_var[varidx] = 2
        CEs = []
        start_time = time.time()

        # hyperparameter in the Proto method, kappa: address probability=1
        # restrict method to not update this hpp
        for i, x in tqdm(enumerate(self.test_instances)):
            kaps = np.concatenate((np.arange(kap, 1.01, 0.2), np.array([1])))
            # make sure the method is at least as good as the non-robust one: using the same default settings first
            best_cf = self.run_proto_robust_one(x, predict_fn, shape, cat_var, theta, kap=kap)
            if best_cf is None:
                best_bound = -10000
            else:
                found, bound = self.is_robust_raw(x, best_cf)
                if found == 1:
                    CEs.append(best_cf)
                    continue
                best_bound = bound if bound is not None else -10000
            for kappa in kaps:
                with HiddenPrints():
                    this_cf = self.run_proto_robust_one(x, predict_fn, shape, cat_var, theta, kap=kappa)
                if this_cf is None:
                    continue
                found, bound = self.is_robust_raw(x, this_cf)
                if bound is None:
                    continue
                if bound >= best_bound:
                    best_cf = this_cf
                    best_bound = bound
                    if found == 1:
                        break
            CEs.append(best_cf)

        print("total computation time in s:", time.time() - start_time)
        # print("guaranteed robustness cf:", guaranteed_rob_count / len(CEs))
        assert len(CEs) == len(self.test_instances)
        return CEs

    def run_proto_robust_one(self, x, predict_fn, shape, cat_var, theta=0., kap=0.1):
        if len(self.discrete_features.keys()) == 0 and len(self.ordinal_features.keys()) == 0:
            cf = cfproto.CounterFactualProto(predict_fn, shape, use_kdtree=True, theta=theta, kappa=kap,
                                             feature_range=(np.array(self.X1.values.min(axis=0)).reshape(1, -1),
                                                            np.array(self.X1.values.max(axis=0)).reshape(1, -1)))
            cf.fit(self.X1.values, trustscore_kwargs=None)
        else:
            cf = cfproto.CounterFactualProto(predict_fn, shape, use_kdtree=True, theta=theta, kappa=kap,
                                             feature_range=(
                                                 np.array(self.X1.min(axis=0)).reshape(1, -1),
                                                 np.array(self.X1.max(axis=0)).reshape(1, -1)),
                                             cat_vars=cat_var,
                                             ohe=False)
            cf.fit(self.X1.values)
        this_point = x
        with HiddenPrints():
            explanation = cf.explain(this_point.reshape(1, -1), Y=None, target_class=None, k=20, k_type='mean',
                                     threshold=0., verbose=True, print_every=100, log_every=100)
        if explanation is None:
            return None
        if explanation["cf"] is None:
            return None
        proto_cf = explanation["cf"]["X"]
        proto_cf = proto_cf[0]
        return np.array(proto_cf)

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

    def run_wachter_robust(self, lam_init=0.1, target_proba=0.6):
        CEs = []
        data_point = np.array(self.X1.values[1])
        shape = (1,) + data_point.shape[:]
        predict_fn = lambda x: self.clf.predict_proba(x)
        start_time = time.time()
        for i, x in tqdm(enumerate(self.test_instances)):
            # make sure the method is at least as good as the non-robust one: using the same default settings first
            best_cf = self.run_wachter_robust_one(x, predict_fn, shape, lam=lam_init, lam_step=10,
                                                  target_proba=target_proba)
            if best_cf is None:
                best_bound = -10000
            else:
                found, bound = self.is_robust_raw(x, best_cf)
                if found == 1:
                    CEs.append(best_cf)
                    continue
                best_bound = bound if bound is not None else -10000
            probs = np.concatenate((np.arange(target_proba, 1.01, 0.1), np.array([1])))
            found_flag = 0
            for prob in probs:
                lambs = [0.01, 0.05, 0.1, 0.2]
                for lamb in lambs:
                    this_cf = self.run_wachter_robust_one(x, predict_fn, shape, lam=lamb, lam_step=10,
                                                          target_proba=prob)
                    if this_cf is None:
                        continue
                    found, bound = self.is_robust_raw(x, this_cf)
                    if bound is None:
                        continue
                    if bound >= best_bound:
                        best_cf = this_cf
                        best_bound = bound
                        if found == 1:
                            found_flag = 1
                if found_flag:  # check found for at each prob, trying 4 lambs
                    break
            CEs.append(best_cf)

        print("total computation time in s:", time.time() - start_time)
        # print("guaranteed robustness cf:", guaranteed_rob_count / len(CEs))
        assert len(CEs) == len(self.test_instances)
        return CEs

    def run_wachter_robust_one(self, x, predict_fn, shape, lam=0.1, lam_step=10, target_proba=0.6):
        cf = Counterfactual(predict_fn, shape, distance_fn='l1', target_proba=target_proba,
                            target_class='other', max_iter=1000, early_stop=50, lam_init=lam,
                            max_lam_steps=lam_step, tol=0.05, learning_rate_init=0.1,
                            feature_range=(0, 1), eps=0.01, init='identity',
                            decay=True, write_dir=None, debug=False)
        with HiddenPrints():
            explanation = cf.explain(x.reshape(1, -1))
        if explanation is None:
            return None
        if explanation["cf"] is None:
            return None
        proto_cf = explanation["cf"]["X"]
        proto_cf = proto_cf[0]
        return np.array(proto_cf)
