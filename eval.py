import torch
import torch.nn.functional as F
import pickle
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
import json
import warnings

from models.inn import Inn
from utils import cfx
from utils.cfx_evaluator import CFXEvaluator
from utils.utilities import seed_everything
from train import prepare_data_and_model

'''
Evaluate the robustness of counterfactual explanations
'''

warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def create_CFX(args, model, minmax, train_data, test_data):
    @torch.no_grad()
    def predictor(X: np.ndarray) -> np.ndarray:
        X = torch.as_tensor(X, dtype=torch.float32)
        with torch.no_grad():
            ret = model.forward_point_weights_bias(X)
            ret = F.softmax(ret, dim=1)
            ret = ret.cpu().numpy()
        return ret

    cfx_generator = cfx.CFX_Generator(predictor, train_data, num_layers=None)

    if args.cfx == 'wachter':
        cfx_x, is_cfx = cfx_generator.run_wachter(scaler=minmax, max_iter=args.wachter_max_iter,
                                                  test_instances=test_data.X, lam_init=args.wachter_lam_init,
                                                  max_lam_steps=args.wachter_max_lam_steps)
    elif args.cfx == 'proto':
        cfx_x, is_cfx = cfx_generator.run_proto(scaler=minmax, theta=args.proto_theta, onehot=args.onehot,
                                                test_instances=test_data.X)
    elif args.cfx == 'counternet':
        with torch.no_grad():
            cfx_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
            cfx_new_list = []
            is_cfx_new_list = []
            for X, y, _ in cfx_dataloader:
                cfx_new, pred = model.forward(X, hard=True)
                is_cfx_new = model.forward_point_weights_bias(cfx_new).argmax(dim=1) == 1 - pred.argmax(dim=1)
                cfx_new_list.append(cfx_new)
                is_cfx_new_list.append(is_cfx_new)

            cfx_x = torch.cat(cfx_new_list, dim=0)
            is_cfx = torch.cat(is_cfx_new_list, dim=0)
    else:
        raise NotImplementedError

    return cfx_x, is_cfx


def main(args):
    seed_everything(args.seed)
    ret = prepare_data_and_model(args)
    train_data, test_data, model, minmax = ret["train_data"], ret["test_data"], ret["model"], ret["minmax"]
    if args.num_to_run is not None:
        test_data.X = test_data.X[:args.num_to_run]
        test_data.y = test_data.y[:args.num_to_run]

    model.load(os.path.join(args.save_dir, args.model))
    model.eval()

    if args.cfx == "counternet":
        inn = Inn.from_IBPModel(model.encoder_net_ori)
    else:
        inn = Inn.from_IBPModel(model.ori_model)
    inn.act = args.config["act"]

    if not os.path.exists(args.cfx_filename):
        print("did not find ", args.cfx_filename)
        cfx_x, is_cfx = create_CFX(args, model, minmax, train_data, test_data)
        with open(args.cfx_filename, 'wb') as f:
            pickle.dump((cfx_x, is_cfx), f)
    else:
        with open(args.cfx_filename, 'rb') as f:
            cfx_x, is_cfx = pickle.load(f)
    if not args.generate_only:
        orig_output = model.forward_point_weights_bias(torch.tensor(test_data.X).float()).argmax(dim=1)
        cfx_output = model.forward_point_weights_bias(cfx_x).argmax(dim=1)

        is_real_cfx = torch.where(torch.tensor(is_cfx))  # ignore indices that failed
        # and filter out indices that don't satisfy f(x) != f(cfx)
        # is_real_cfx = torch.where(orig_output[is_real_cfx] != cfx_output[is_real_cfx])

        if not torch.all(orig_output[is_real_cfx] != cfx_output[is_real_cfx]):
            print("\n\n\n\n\n\nproblem, some fake CFX included")
            print(torch.where(orig_output[is_real_cfx] == cfx_output[is_real_cfx]))
            print("\n\n\n\n\n\n")

        cfx_eval = CFXEvaluator(cfx_x, is_cfx, model.encoder_verify if args.cfx == "counternet" else model, None,
                                train_data, test_data, inn, args.log_filename)
        cfx_eval.log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help="model name (don't include path or .pt)")
    parser.add_argument('--config', type=str, default="assets/german_credit.json",
                        help='config file for the dataset and the model')
    parser.add_argument('--save_dir', type=str, default="trained_models", help="directory where model is saved")
    parser.add_argument('--cfx_save_dir', type=str, default="saved_cfxs", help="directory to save cfx to")
    parser.add_argument('--log_save_dir', type=str, default="logs", help="directory to save logs to")
    parser.add_argument('--cfx_filename', default=None,
                        help="name of the file where CFX are or should be stored. If blank, use log_name")
    parser.add_argument('--log_name', type=str, default=None, help="name of log file, end with .txt")
    parser.add_argument('--cfx', type=str, default="wachter", choices=["wachter", "proto", "counternet"])
    parser.add_argument('--num_to_run', type=int, default=None, help='number of test examples to run')

    parser.add_argument('--onehot', action='store_true', help='whether to use one-hot encoding')
    parser.add_argument('--proto_theta', type=float, default=100, help='theta for proto')
    parser.add_argument('--wachter_max_iter', type=int, default=100, help='max iter for wachter')
    parser.add_argument('--wachter_lam_init', type=float, default=1e-3, help='initial lambda for wachter')
    parser.add_argument('--wachter_max_lam_steps', type=int, default=10, help='max lambda steps for wachter')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--generate_only', action='store_true',
                        help='if true, only generate and save cfx, do not eval robustness')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        args.config = json.load(f)
    if not os.path.exists(args.cfx_save_dir):
        os.makedirs(args.cfx_save_dir)
    if not os.path.exists(args.log_save_dir):
        os.makedirs(args.log_save_dir)

    if args.log_name is None:
        args.log_name = args.model + ".txt"
    if args.cfx_filename is None:
        args.cfx_filename = args.log_name[:-4]
    args.log_filename = os.path.join(args.log_save_dir, args.log_name)
    args.cfx_filename = os.path.join(args.cfx_save_dir, args.cfx_filename)

    main(args)
