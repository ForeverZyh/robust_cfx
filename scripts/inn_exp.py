import argparse
import json
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import torch

from train import prepare_data_and_model
from inn.utilexp import UtilExp
from models.inn import Inn

'''
Given a set of 10 models trained on the same dataset with different seeds, 
and the associated CFX, measure the validity of each set of CFX across the
10 models. Only look at validity (pointwise predictions), not epsilon robustness.

Saves a csv file with the validity rate for the CFXs across all samples and across
samples that had a valid CFX.

Currently only used for ROAR, in theory, could be extended to other CFX generation
techniques that work on INN models (rather than tensorflow or CounterNet models)
'''

warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def eval(args, models, all_preds, cfxs, is_cfxs):
    all_data = []
    l2_norms = []
    l2_norm_normeds = []
    for i in range(args.model_cnt):
        for k, param in enumerate(models[i].parameters()):
            l2_norm = torch.norm(param).item()
            l2_norm_normed = np.sqrt(l2_norm ** 2 / param.numel())
            if k < len(l2_norms):
                l2_norms[k].append(l2_norm)
                l2_norm_normeds[k].append(l2_norm_normed)
            else:
                l2_norms.append([l2_norm])
                l2_norm_normeds.append([l2_norm_normed])
        for j in range(args.model_cnt):
            if i != j:
                if i < j and args.verbose:
                    print(f"==={i} vs. {j}")
                    print(models[i].difference(models[j]))
                this_model = models[i]
                these_preds = all_preds[i]
                these_cfx = torch.tensor(cfxs[j])
                these_is_cfx = torch.tensor(is_cfxs[j])

                cfx_preds = this_model.forward_point_weights_bias(these_cfx.float()).argmax(dim=1)

                is_valid = torch.where(these_is_cfx, these_preds != cfx_preds, torch.tensor([False]))
                is_valid_pct = torch.sum(is_valid).item() / max(1, torch.sum(these_is_cfx).item())
                is_valid_overall_pct = torch.sum(is_valid).item() / max(1, len(is_valid))

                data = [is_valid_pct, is_valid_overall_pct]
                all_data.append(data)

    l2_norms = np.mean(np.array(l2_norms), axis=-1)
    l2_norm_normeds = np.mean(np.array(l2_norm_normeds), axis=-1)
    if args.verbose:
        print(l2_norms, l2_norm_normeds)
    df = pd.DataFrame(all_data, columns=['validity_all', 'validity_for_cfx'])
    df.to_csv(os.path.join(args.log_dir, args.model + "_" + args.cfx_technique + ".csv"),
              index=False)


def main(args):
    models = []
    cfxs = []
    is_cfxs = []
    all_preds = []
    orig_model_name = args.model
    for i in args.target_idxs:
        args.model = orig_model_name + str(i)
        ret = prepare_data_and_model(args)
        train_data, test_data, model, minmax, preprocessor = ret["train_data"], ret["test_data"], ret["model"], ret[
            "minmax"], ret["preprocessor"]

        if args.num_to_run is not None:
            test_data.X = test_data.X[:args.num_to_run]
            test_data.y = test_data.y[:args.num_to_run]

        model.load(os.path.join(args.save_dir, args.model))
        model.eval()
        models.append(model)

        preds = model.forward_point_weights_bias(torch.tensor(test_data.X).float()).argmax(dim=-1)
        all_preds.append(preds)

        # load cfx
        cfx_filename = os.path.join(args.cfx_dir, args.model + "_" + args.cfx_technique)
        if not os.path.exists(cfx_filename) or args.force_regen_cfx:
            util_exp = UtilExp(model, preprocessor, train_data.X, test_data.X, test_data.y, train_data,
                               target_p=args.target_p, target_s=args.target_s)

            util_exp.inn_delta_non_0 = Inn.from_IBPModel(model.encoder_net_ori)
            util_exp.inn_delta_non_0.act = args.config["act"]
            cfx_x, is_cfx = util_exp.run_ROAR(labels=preds.numpy())

            with open(cfx_filename, 'wb') as f:
                pickle.dump((cfx_x, is_cfx), f)

        with open(cfx_filename, 'rb') as f:
            cfx_x, is_cfx = pickle.load(f)
            cfxs.append(np.array(cfx_x))
            is_cfxs.append(np.array(is_cfx))

    if not args.generate_only:
        eval(args, models, all_preds, cfxs, is_cfxs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model name with final number (0, 1, etc.) omitted")
    parser.add_argument("dataset")

    parser.add_argument('--cfx_technique', type=str, default="roar", help="cfx technique - default roar because that's used here")
    parser.add_argument("--cfx_dir", default="saved_cfxs", help="directory where cfxs are saved")
    parser.add_argument("--save_dir", default='trained_models',
                        help="directory where models are saved, if omitted will be trained_models/dataset")
    parser.add_argument('--log_dir', type=str, default='logs', help='directory to save logs')
    parser.add_argument('--force_regen_cfx', action='store_true',
                        help='whether to force regenerate cfx if the file exists')
    parser.add_argument('--generate_only', action='store_true', help='whether to only generate cfx')
    parser.add_argument('--target_idxs', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], nargs='+', type=int,
                        help='which random seeds to run on')
    parser.add_argument('--num_to_run', type=int, default=None)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--finetune', action='store_true')

    # experimental parameters - not used in paper
    parser.add_argument('--target_p', type=float, default=0, help="target proximity for ROAR")
    parser.add_argument('--target_s', type=float, default=0, help="target sparsity for ROAR")


    args = parser.parse_args()
    args.model_cnt = len(args.target_idxs)

    args.config = f'assets/{args.dataset}.json'
    with open(args.config, 'r') as f:
        args.config = json.load(f)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    args.remove_pct = None
    main(args)
