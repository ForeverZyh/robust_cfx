import argparse
import os
import pandas as pd

# if possible, import function that will map file names to data about the run
try:
    from get_chtc_mapping import get_mapping_reverse
except:
    pass

'''
Given logs (as generated by eval.py) extract the relevant data 
and save it to a CSV file.
'''

def get_dataset(file):
    if 'german' in file:
        return 'german'
    elif 'heloc' in file:
        return 'heloc'
    elif 'ctg' in file:
        return 'ctg'
    elif "student" in file:
        return "student"
    elif "taiwan" in file:
        return "taiwan"
    else:
        print("error: can't identify dataset")
        return -1

def get_training_method(file):
    if 'Standard' in file:
        return 'Standard'
    elif 'IBP' in file:
        return 'IBP'
    else:
        print("error: can't identify model training method")
        return -1
    
def get_cfx(file, dataset):
    if 'wachter' in file:
        return 'wachter'
    elif 'proto' in file:
        return 'proto'
    elif 'counternetoursroar' in file:
        return 'counternetoursroar'
    elif 'counternetinn' in file:
        return 'counternetinn'
    elif 'counternetours' in file:
        return 'counternetours'
    elif 'counternetibp' in file:
        return 'counternetibp'
    elif 'counternetcrownibp' in file:
        return 'counternetcrownibp'
    elif 'counternet' in file:
        return 'counternet'
    else:
        try:
            if "_" in file:
                chtc_num = get_mapping_reverse(file.split(dataset)[1].split("_")[0])
            elif "e100" in file:
                chtc_num = get_mapping_reverse(file.split(dataset)[1].split("e100")[0])
            else:
                chtc_num = get_mapping_reverse(file.split(dataset)[1].split(".")[0])
            if chtc_num[3] == 'ours':
                return 'counternetours'
            elif chtc_num[3] == 'ibp':
                return 'counternetibp'
            elif chtc_num[3] == 'crownibp':
                return 'counternetcrownibp'
            else:
                return 'counternet'
        except:
            print("error: can't identify CFX generation method")
            return -1
        
def get_epoch_eps_r(file, dataset):
    try:
        chtc_num = get_mapping_reverse(file.split(dataset)[1].split("_")[0])
        return chtc_num[2], chtc_num[4], chtc_num[5]
    except:
        return 0, 0, 0
    
def main(args):
    all_data = []
    for file in os.listdir(args.log_dir):
        if os.path.isdir(args.log_dir + "/" + file):
            continue
        if file[0] == ".":
            continue

        try:
            with open(args.log_dir + "/" + file) as f:
                lines = f.readlines()
        except:
            print("skipping file",file)
            continue

        if args.verbose:
            print(lines)

        if lines == []:
            continue

        dataset = get_dataset(file)
        if args.target_datasets != [] and dataset not in args.target_datasets:
            continue

        training_method = get_training_method(file)
        cfx = get_cfx(file, dataset)
        epoch, eps, r = get_epoch_eps_r(file, dataset) # ran multiple epochs for student & taiwan

        # epsilon = float(lines[0].split(" ")[-1].split("\n")[0])
        # bias_epsilon = float(lines[1].split(" ")[-1].split("\n")[0])
        try:
            lines = ["", ""] + lines
            test_acc = float(lines[2].split(" ")[-2].split("%")[0])/100
            train_acc = float(lines[3].split(" ")[-2].split("%")[0])/100
            validity = float(lines[4].split(" ")[-2].split("%")[0])/100
            robustness_us = float(lines[5].split(" ")[-2].split("%")[0])/100
            if not args.skip_milp:
                robustness_mlp = float(lines[6].split(" ")[-2].split("%")[0])/100
                bounds_better = int(lines[7].split(" ")[0])
                total_samples_w_cfx = int(lines[4].split("/")[1].split(")")[0])
                bounds_better_frac = bounds_better/total_samples_w_cfx
                proximity = float(lines[8].split(" ")[-3].split(",")[0])
                proximity_std = float(lines[8].split(" ")[-1].split("\n")[0])
                sparsity = float(lines[9].split(" ")[-3].split(",")[0])
                sparsity_std = float(lines[9].split(" ")[-1].split("\n")[0])
                data_manifold_dist = float(lines[10].split(" ")[-3].split(",")[0])
                data_manifold_dist_std = float(lines[10].split(" ")[-1].split("\n")[0])
            else:
                robustness_mlp, bounds_better, total_samples_w_cfx, bounds_better_frac = 0, 0, 0, 0
                proximity = float(lines[6].split(" ")[-3].split(",")[0])
                proximity_std = float(lines[6].split(" ")[-1].split("\n")[0])
                sparsity = float(lines[7].split(" ")[-3].split(",")[0])
                sparsity_std = float(lines[7].split(" ")[-1].split("\n")[0])
                data_manifold_dist = float(lines[8].split(" ")[-3].split(",")[0])
                data_manifold_dist_std = float(lines[8].split(" ")[-1].split("\n")[0])
        except:
            continue

        if args.verbose:
            # print("eps: ", epsilon)
            # print("bias_eps: ", bias_epsilon)
            print("test_acc: ", test_acc)
            print("train_acc: ", train_acc)
            print("validity: ", validity)
            print("robustness_us: ", robustness_us)
            print("robustness_mlp: ", robustness_mlp)
            print("bounds_better: ", bounds_better_frac)
            print("proximity: ", proximity)
            print("proximity_std: ", proximity_std)
            print("sparsity: ", sparsity)
            print("sparsity_std: ", sparsity_std)
            print("data_manifold_dist: ", data_manifold_dist)
            print("data_manifold_dist_std: ", data_manifold_dist_std)
        data = [dataset, training_method, cfx, test_acc, train_acc, validity,
                robustness_us, robustness_mlp, bounds_better_frac, proximity, proximity_std, 
                sparsity, sparsity_std, data_manifold_dist, data_manifold_dist_std, epoch,
                eps, r]
        all_data.append(data)


    columns = ['dataset', 'training_method', 'cfx', 'test_acc',
                'train_acc', 'validity', 'robustness_us', 'robustness_mlp', 'bounds_better_frac', 
                'proximity', 'proximity_std', 'sparsity', 'sparsity_std', 'data_manifold_dist', 
                'data_manifold_dist_std', 'epoch', 'eps', 'ratio']
    df = pd.DataFrame(all_data, columns=columns)
    df.to_csv("all_data_robust.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default="logs", help="directory where logs are saved")
    parser.add_argument('--verbose', action='store_true', help='print out all data')
    parser.add_argument('--skip_milp', action='store_true', help='skip MILP')
    parser.add_argument('--target_datasets', default = [], nargs='+', help='datasets to run on')

    args = parser.parse_args()
    main(args)