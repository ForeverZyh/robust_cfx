import argparse
import json
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import torch

from train import prepare_data_and_model
try:
    from get_chtc_mapping import get_roar_mapping, get_chtc_num
except:
    pass

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
    for i in range(args.model_cnt):
        try:
            chtcnum = get_chtc_num(args.model_type, args.dataset, args.epoch, args.cfx_technique, args.eps, args.r)
        except:
            chtcnum = ""
        args.model = args.model_type + args.dataset + chtcnum + "_" + str(i)
        ret = prepare_data_and_model(args)
        train_data, test_data, model, minmax = ret["train_data"], ret["test_data"], ret["model"], ret["minmax"]
        model.load(os.path.join(args.model_dir, args.model))
        model.eval()
        models.append(model)

        preds = model.forward_point_weights_bias(torch.tensor(test_data.X).float()).argmax(dim=-1)
        all_preds.append(preds)

        # load cfx
        if args.cfx_technique == 'roar':
            try:
                basefile = get_roar_mapping(chtcnum)
            except:
                basefile = args.model_type + args.dataset + "_"
        else:
            basefile = args.model_type + args.dataset + chtcnum + "_"
        cfx_filename = os.path.join(args.cfx_dir, basefile + str(i))
        with open(cfx_filename, 'rb') as f:
            cfx_x, is_cfx = pickle.load(f)
            cfxs.append(torch.tensor(cfx_x))
            is_cfxs.append(torch.tensor(is_cfx))
        print(torch.tensor(cfx_x).shape)

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
                these_cfx = cfxs[j]
                these_is_cfx = is_cfxs[j]


                print(cfxs[j].shape)
                cfx_preds = this_model.forward_point_weights_bias(these_cfx.float()).argmax(dim=1)

                is_valid = torch.where(these_is_cfx, these_preds != cfx_preds, torch.tensor([False]))
                is_valid_pct = torch.sum(is_valid).item() / torch.sum(these_is_cfx).item()
                is_valid_overall_pct = torch.sum(is_valid).item() / len(is_valid)

                data = [is_valid_pct, is_valid_overall_pct]
                all_data.append(data)

    l2_norms = np.mean(np.array(l2_norms), axis=-1)
    l2_norm_normeds = np.mean(np.array(l2_norm_normeds), axis=-1)
    if args.verbose:
        print(l2_norms, l2_norm_normeds)
    df = pd.DataFrame(all_data, columns=['validity_all', 'validity_for_cfx'])
    if not os.path.exists(os.path.join("logs", "validity")):
        os.makedirs(os.path.join("logs", "validity"))
    df.to_csv(os.path.join("logs", "validity", args.model_type + args.dataset + chtcnum + "e" + str(args.epoch) \
                            + "eps" + str(args.eps) + "r" + str(args.r) + ".csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("model_type", help="Standard or IBP", choices=["Standard", "IBP"])
    parser.add_argument("cfx_technique", help="ours, ibp, crownibp, or none")
    parser.add_argument("--cfx_dir", default="saved_cfxs", help="directory where cfxs are saved")
    parser.add_argument("--model_dir", default='trained_models',
                        help="directory where models are saved, if omitted will be trained_models")
    parser.add_argument('--onehot', action='store_true', help='whether to use one-hot encoding')
    parser.add_argument('--model_cnt', type=int, default=10, help='how many models trained for each dataset')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument('--eps', type=float, default=0.2)
    parser.add_argument('--r', type=float, default=0.05)
    # parser.add_argument('--seed', default=0)

    # parser.add_argument('--epsilon', type=float, default=1e-2, help='epsilon for IBP')
    # parser.add_argument('--bias_epsilon', type=float, default=1e-3, help='bias epsilon for IBP')

    args = parser.parse_args()
    args.cfx = "counternet"
    if args.cfx_technique != "none":
        args.cfx += args.cfx_technique
    if args.dataset == 'german':
        args.config = 'assets/german_credit.json'
    else:
        args.config = 'assets/' + args.dataset + '.json'

    with open(args.config, 'r') as f:
        args.config = json.load(f)

    args.remove = []

    main(args)
