import argparse
import json
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import torch

from train import prepare_data_and_model

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
    orig_model_name = args.model
    models, orig_models = [], [] 
    cfxs = []
    is_cfxs = []
    all_preds = []
    for i in range(args.model_cnt):
        args.model = orig_model_name + str(i)
        if args.finetune:
            ret = prepare_data_and_model(args)
            _, test_data, orig_model = ret["train_data"], ret["test_data"], ret["model"]
            orig_model.load(os.path.join(args.save_dir, args.model))
            orig_model.eval()
            orig_models.append(orig_model)
            args.model += "_finetune"
        ret = prepare_data_and_model(args)
        _, test_data, model = ret["train_data"], ret["test_data"], ret["model"]
        model.load(os.path.join(args.save_dir, args.model))
        model.eval()
        models.append(model)

        if args.finetune:
            preds = orig_model.forward_point_weights_bias(torch.tensor(test_data.X).float()).argmax(dim=-1)
        else:
            preds = model.forward_point_weights_bias(torch.tensor(test_data.X).float()).argmax(dim=-1)
        all_preds.append(preds)

        # load cfx
        cfx_filename = os.path.join(args.cfx_dir, args.model + str(i) + "_" + args.cfx_technique)

        with open(cfx_filename, 'rb') as f:
            cfx_x, is_cfx = pickle.load(f)
            cfxs.append(torch.tensor(cfx_x))
            is_cfxs.append(torch.tensor(is_cfx))

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
        if args.finetune:
            this_model = models[i]
            these_preds = all_preds[i]
            these_cfx = cfxs[i]
            these_is_cfx = is_cfxs[i]

            cfx_preds = this_model.forward_point_weights_bias(these_cfx.float()).argmax(dim=1)

            is_valid = torch.where(these_is_cfx, these_preds != cfx_preds, torch.tensor([False]))
            if torch.sum(these_is_cfx).item() == 0:
                is_valid_pct = 0
                is_valid_overall_pct = 0
            else:
                is_valid_pct = torch.sum(is_valid).item() / torch.sum(these_is_cfx).item()
                is_valid_overall_pct = torch.sum(is_valid).item() / len(is_valid)

            data = [is_valid_pct, is_valid_overall_pct]
            all_data.append(data)
        else:
            for j in range(args.model_cnt):
                if i != j:
                    if i < j and args.verbose:
                        print(f"==={i} vs. {j}")
                        print(models[i].difference(models[j]))
                    # model and preds comes from original model, cfxs come from finetuned model
                    this_model = models[i]
                    these_preds = all_preds[i]
                    these_cfx = cfxs[j]
                    these_is_cfx = is_cfxs[j]

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

    df.to_csv(os.path.join(args.log_save_dir, args.model + "_" + args.cfx_technique + ".csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model name with final number (0, 1, etc.) omitted")
    parser.add_argument("dataset")
    parser.add_argument("cfx_technique", help="ours, ibp, crownibp, roar, or none")
    parser.add_argument("--cfx_dir", default="saved_cfxs", help="directory where cfxs are saved")
    parser.add_argument("--save_dir", default='trained_models',
                        help="directory where models are saved, if omitted will be trained_models")
    parser.add_argument('--model_cnt', type=int, default=10, help='how many models trained for each dataset')
    parser.add_argument('--log_save_dir', default='logs/validity')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--finetune', action='store_true')

    args = parser.parse_args()

    if not os.path.exists(args.log_save_dir):
        os.makedirs(args.log_save_dir)

    args.config = 'assets/' + args.dataset + '.json'
    with open(args.config, 'r') as f:
        args.config = json.load(f)

    args.remove_pct = None

    main(args)
