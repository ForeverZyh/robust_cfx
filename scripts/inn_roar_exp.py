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

'''
Given a set of 10 models trained on the same dataset with different seeds, 
and the associated CFX, measure the validity of each set of CFX across the
10 models. Only look at validity (pointwise predictions), not epsilon robustness.

Saves a csv file with the validity rate for the CFXs across all samples and across
samples that had a valid CFX.
'''

warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main(args):
    models = []
    cfxs = []
    is_cfxs = []
    all_preds = []
    for i in range(args.model_cnt):
        args.model = args.model_type + args.dataset + args.cfx + str(i)
        ret = prepare_data_and_model(args)
        train_data, test_data, model, minmax, preprocessor = ret["train_data"], ret["test_data"], ret["model"], ret[
            "minmax"], ret["preprocessor"]
        model.load(os.path.join(args.model_dir, args.model))
        model.eval()
        models.append(model)

        preds = model.forward_point_weights_bias(torch.tensor(test_data.X).float()).argmax(dim=-1)
        all_preds.append(preds)

        # load cfx
        cfx_filename = os.path.join(args.cfx_dir, args.model_type + args.dataset + args.cfx + "inn" + str(i))
        if not os.path.exists(cfx_filename) or args.force_regen_cfx:
            util_exp = UtilExp(model, preprocessor, train_data.X, test_data.X)
            util_exp.inn_delta_non_0 = model.encoder_net_ori.to_Inn()
            cfx_x, is_cfx = util_exp.run_ROAR(labels=preds.numpy())
            # util_exp.evaluate_ces([x if y else None for x, y in zip(cfx_x, is_cfx)])
            with open(cfx_filename, 'wb') as f:
                pickle.dump((cfx_x, is_cfx), f)

        with open(cfx_filename, 'rb') as f:
            cfx_x, is_cfx = pickle.load(f)
            cfxs.append(np.array(cfx_x))
            is_cfxs.append(np.array(is_cfx))

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
                if i < j:
                    print(f"==={i} vs. {j}")
                    print(models[i].difference(models[j]))
                this_model = models[i]
                these_preds = all_preds[i]
                these_cfx = torch.tensor(cfxs[j])
                these_is_cfx = torch.tensor(is_cfxs[j])

                cfx_preds = this_model.forward_point_weights_bias(these_cfx.float()).argmax(dim=1)

                is_valid = torch.where(these_is_cfx, these_preds != cfx_preds, torch.tensor([False]))
                is_valid_pct = torch.sum(is_valid).item() / torch.sum(these_is_cfx).item()
                is_valid_overall_pct = torch.sum(is_valid).item() / len(is_valid)

                data = [is_valid_pct, is_valid_overall_pct]
                all_data.append(data)

    l2_norms = np.mean(np.array(l2_norms), axis=-1)
    l2_norm_normeds = np.mean(np.array(l2_norm_normeds), axis=-1)
    print(l2_norms, l2_norm_normeds)
    df = pd.DataFrame(all_data, columns=['validity_all', 'validity_for_cfx'])
    if not os.path.exists(os.path.join("logs", "validity")):
        os.makedirs(os.path.join("logs", "validity"))
    df.to_csv(os.path.join("logs", "validity", args.model_type + args.dataset + args.cfx + "inn.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("model_type", help="Standard or IBP", choices=["Standard", "IBP"])
    parser.add_argument('--cfx', type=str, default="counternet", help="only counternet model is trained for now",
                        choices=["counternet"])
    parser.add_argument("--cfx_dir", default="saved_cfxs/fromchtc", help="directory where cfxs are saved")
    parser.add_argument("--model_dir", default='trained_models',
                        help="directory where models are saved, if omitted will be trained_models/dataset")
    parser.add_argument('--onehot', action='store_true', help='whether to use one-hot encoding')
    parser.add_argument('--model_cnt', type=int, default=10, help='how many models trained for each dataset')
    parser.add_argument('--robust', action='store_true', help='whether to use INN on cfx generators')
    parser.add_argument('--force_regen_cfx', action='store_true',
                        help='whether to force regenerate cfx if the file exists')

    args = parser.parse_args()

    if args.dataset == 'german':
        args.config = 'assets/german_credit.json'
    elif args.dataset == 'heloc':
        args.config = 'assets/heloc.json'
    elif args.dataset == 'ctg':
        args.config = 'assets/ctg.json'
    elif args.dataset == 'taiwan':
        args.config = 'assets/taiwan.json'
    elif args.dataset == 'student':
        args.config = 'assets/student.json'
    with open(args.config, 'r') as f:
        args.config = json.load(f)

    main(args)
