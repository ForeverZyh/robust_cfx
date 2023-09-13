import torch
import torch.nn.functional as F
import pickle
import numpy as np
import os

import argparse

from models.inn import Inn
from models.IBPModel import FNN
from utils import cfx
from utils import dataset
from utils.cfx_evaluator import CFXEvaluator

'''
Evaluate the robustness of counterfactual explanations
'''


def create_CFX(args, model, minmax, train_data, test_data, num_hiddens):
    @torch.no_grad()
    def predictor(X: np.ndarray) -> np.ndarray:
        X = torch.as_tensor(X, dtype=torch.float32)
        with torch.no_grad():
            ret = model.forward_point_weights_bias(X)
            ret = F.softmax(ret, dim=1)
            ret = ret.cpu().numpy()
        return ret

    cfx_generator = cfx.CFX_Generator(predictor, train_data, num_layers=len(num_hiddens))

    if args.cfx == 'wachter':
        cfx_x, is_cfx = cfx_generator.run_wachter(scaler=minmax, max_iter=args.wachter_max_iter,
                                                  test_instances=test_data.X, lam_init=args.wachter_lam_init,
                                                  max_lam_steps=args.wachter_max_lam_steps)
    else:
        cfx_x, is_cfx = cfx_generator.run_proto(scaler=minmax, theta=args.proto_theta, onehot=args.onehot,
                                                test_instances=test_data.X)

    return cfx_x, is_cfx


def main(args):
    torch.random.manual_seed(0)

    if args.cfx == 'proto':
        feature_types = dataset.CREDIT_FEAT_PROTO
    else:
        feature_types = dataset.CREDIT_FEAT

    if args.onehot:
        minmax = None
        train_data, min_vals, max_vals = dataset.load_data("data/german_train.csv", "credit_risk", feature_types)
        test_data, _, _ = dataset.load_data("data/german_test.csv", "credit_risk", feature_types, min_vals, max_vals)
    else:
        train_data, test_data, minmax = dataset.load_data_v1("data/german_train.csv", "data/german_test.csv",
                                                             "credit_risk",
                                                             feature_types)

    if args.num_to_run is not None:
        test_data.X = test_data.X[:args.num_to_run]
        test_data.y = test_data.y[:args.num_to_run]

    dim_in = train_data.num_features

    num_hiddens = [10, 10]

    model = FNN(dim_in, 2, num_hiddens, epsilon=args.epsilon, bias_epsilon=args.bias_epsilon)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, args.model)))
    model.eval()
    inn = Inn.from_IBPModel(model)

    if not os.path.exists(args.cfx_filename):
        cfx_x, is_cfx = create_CFX(args, model, minmax, train_data, test_data, num_hiddens)
        with open(args.cfx_filename, 'wb') as f:
            pickle.dump((cfx_x, is_cfx), f)
    else:
        with open(args.cfx_filename, 'rb') as f:
            cfx_x, is_cfx = pickle.load(f)
    cfx_eval = CFXEvaluator(cfx_x, is_cfx, model, model, train_data, test_data, inn, args.log_filename)
    cfx_eval.log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='path to model with .pt extension')
    parser.add_argument('--save_dir', type=str, default="trained_models", help="directory to save models to")
    parser.add_argument('--cfx_save_dir', type=str, default="saved_cfxs", help="directory to save cfx to")
    parser.add_argument('--log_save_dir', type=str, default="logs", help="directory to save models to")
    parser.add_argument('--log_name', type=str, default=None, help="name of log file")
    parser.add_argument('--cfx', type=str, default="wachter", choices=["wachter", "proto"])
    parser.add_argument('--num_to_run', type=int, default=None, help='number of test examples to run')

    parser.add_argument('--onehot', action='store_true', help='whether to use one-hot encoding')
    parser.add_argument('--proto_theta', type=float, default=100, help='theta for proto')
    parser.add_argument('--wachter_max_iter', type=int, default=100, help='max iter for wachter')
    parser.add_argument('--wachter_lam_init', type=float, default=1e-3, help='initial lambda for wachter')
    parser.add_argument('--wachter_max_lam_steps', type=int, default=10, help='max lambda steps for wachter')

    parser.add_argument('--epsilon', type=float, default=1e-2, help='epsilon for IBP')
    parser.add_argument('--bias_epsilon', type=float, default=1e-3, help='bias epsilon for IBP')
    args = parser.parse_args()

    if not os.path.exists(args.cfx_save_dir):
        os.makedirs(args.cfx_save_dir)
    if not os.path.exists(args.log_save_dir):
        os.makedirs(args.log_save_dir)

    args.log_filename = os.path.join(args.log_save_dir, args.log_name)
    args.cfx_filename = os.path.join(args.cfx_save_dir, args.log_name[:-4])

    main(args)
