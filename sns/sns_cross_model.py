import argparse
import json
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
import copy

from utils.dataset import prepare_data
import models.IBPModel_tf as IBPModel_tf
from utils.utilities import FNNDims

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

'''
Given a set of 10 models trained on the same dataset with different seeds, 
and the associated CFX, measure the validity of each set of CFX across the
10 models. Only look at validity (pointwise predictions), not epsilon robustness.

Saves a csv file with the validity rate for the CFXs across all samples and across
samples that had a valid CFX.
'''

warnings.filterwarnings("ignore", category=ResourceWarning)

def main(args):
    models = []
    cfxs = []
    is_cfxs = []
    all_preds = []
    num_to_run = args.num_to_run
    for i in range(args.model_cnt):
        # check if CFX file exists!
        cfx_filename = os.path.join(args.cfx_dir, args.model + str(i) + "_sns.npy")
        if not os.path.exists(cfx_filename):
            print(f"CFX file {cfx_filename} does not exist. Skipping...")
            continue
        
        ret = prepare_data(args)
        X_train = np.array(ret["train_data"].X).astype(np.float32)
        y_train = np.array(ret["train_data"].y)
        X_test = np.array(ret["test_data"].X).astype(np.float32)
        y_test = np.array(ret["test_data"].y)
        preprocessor = ret["preprocessor"]

        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)

        if args.config["act"] == 0:
            act = tf.keras.activations.relu
        elif args.config["act"] > 0:
            act = lambda: tf.keras.layers.LeakyReLU(args.config["act"])
        dim_in = X_train.shape[1]
        enc_dims = FNNDims(dim_in, args.config["encoder_dims"])        
        pred_dims = FNNDims(None, args.config["decoder_dims"])
        exp_dims = FNNDims(None, args.config["explainer_dims"])

        model = IBPModel_tf.CounterNet(enc_dims, pred_dims, exp_dims, 2,
                                    epsilon_ratio=args.config["eps_ratio"],
                                    activation=act, dropout=args.config["dropout"], preprocessor=None,
                                    config=args.config)

        model.build()
        if args.finetune:
            model.load(os.path.join(args.model_dir, args.model + str(i) + "_finetune"))
        else:
            model.load(os.path.join(args.model_dir, args.model + str(i)))

        model = model.model

        models.append(model)

        if args.finetune:
            dim_in = X_train.shape[1]
            enc_dims = FNNDims(dim_in, args.config["encoder_dims"])        
            pred_dims = FNNDims(None, args.config["decoder_dims"])
            exp_dims = FNNDims(None, args.config["explainer_dims"])
            orig_model = IBPModel_tf.CounterNet(enc_dims, pred_dims, exp_dims, 2,
                                    epsilon_ratio=args.config["eps_ratio"],
                                    activation=act, dropout=args.config["dropout"], preprocessor=None,
                                    config=args.config)
            orig_model.build()
            orig_model.load(os.path.join(args.model_dir, args.model + str(i)))
            orig_model = orig_model.model
            preds = orig_model.predict(X_test[:num_to_run]).argmax(axis=-1)
        else:
            preds = model.predict(X_test[:num_to_run]).argmax(axis=-1)
        all_preds.append(preds)

        with open(cfx_filename, 'rb') as f:
            cfx_x, is_cfx = pickle.load(f)
            cfxs.append(torch.tensor(cfx_x[:num_to_run]))
            is_cfxs.append(torch.tensor(is_cfx[:num_to_run]))

    all_data = []
    l2_norms = []
    l2_norm_normeds = []
    for i in range(len(models)):
        if args.finetune:
            this_model = models[i]
            these_preds = all_preds[i]
            these_cfx = cfxs[i]
            these_is_cfx = is_cfxs[i]
            these_cfx = these_cfx.numpy()

            cfx_preds = this_model.predict(these_cfx).argmax(axis=-1)
            these_preds = torch.tensor(these_preds).bool()
            cfx_preds = torch.tensor(cfx_preds).bool()
            equality = (these_preds != cfx_preds).bool()

            is_valid = torch.where(these_is_cfx.bool(), equality, torch.tensor([False]))
            is_valid_pct = torch.sum(is_valid).item() / torch.sum(these_is_cfx).item()
            is_valid_overall_pct = torch.sum(is_valid).item() / len(is_valid)   

            data = [is_valid_pct, is_valid_overall_pct]
            all_data.append(data)
        else: 
            for j in range(len(models)):
                if i != j:
                    if i < j and args.verbose:
                        print(f"==={i} vs. {j}")
                    this_model = models[i]
                    these_preds = all_preds[i]
                    these_cfx = cfxs[j]
                    these_is_cfx = is_cfxs[j]

                    # convert these_preds (currently a tensor) to numpy
                    these_cfx = these_cfx.numpy()
                    cfx_preds = this_model.predict(these_cfx).argmax(axis=-1)

                    these_preds = torch.tensor(these_preds).bool()
                    cfx_preds = torch.tensor(cfx_preds).bool()

                    equality = (these_preds != cfx_preds).bool()

                    is_valid = torch.where(these_is_cfx.bool(), equality, torch.tensor([False]))
                    if torch.sum(these_is_cfx) == 0:
                        is_valid_pct = 0
                        is_valid_overall_pct = 0
                    else:
                        is_valid_pct = torch.sum(is_valid).item() / torch.sum(these_is_cfx).item()
                        is_valid_overall_pct = torch.sum(is_valid).item() / len(is_valid)

                    data = [is_valid_pct, is_valid_overall_pct]
                    all_data.append(data)

    df = pd.DataFrame(all_data, columns=['validity_all', 'validity_for_cfx'])
    if not os.path.exists(args.log_save_dir):
        os.makedirs(args.log_save_dir)
    df.to_csv(os.path.join(args.log_save_dir, args.model + "_sns.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model",help="model name")
    parser.add_argument("dataset")
    parser.add_argument("technique", help="l1, l2, or pgd", choices=["l1", "l2", "pgd"])
    parser.add_argument("--cfx_dir", default="sns/saved_cfxs", help="directory where cfxs are saved")
    parser.add_argument("--model_dir", default='sns/saved_keras_models', help="directory where models are saved")
    parser.add_argument('--model_cnt', type=int, default=10, help='how many models trained for each dataset')
    parser.add_argument('--log_save_dir', default='sns/logs')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--seed', default=0)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--num_to_run', type=int, default=None)

    args = parser.parse_args()

    args.config = f'assets/{args.dataset}.json'
    with open(args.config, 'r') as f:
        args.config = json.load(f)

    if args.num_to_run is None:
        if args.dataset == 'ctg':
            args.num_to_run = 500
        else:
            args.num_to_run = 1000

    args.remove_pct = None

    main(args)
