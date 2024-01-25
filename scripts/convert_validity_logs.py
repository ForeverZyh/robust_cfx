import argparse
import os
import pandas as pd
import numpy as np

from convert_logs import get_dataset, get_training_method, get_cfx

def main(args):
    all_data = []
    for file in os.listdir(args.log_dir):
        if file[-3:] != 'csv':
            continue
        if file[0] == ".":
            continue

        if 'e100' in file:
            epochs = 100
        else:
            epochs = 200
        if 'eps' in file:
            eps = float(file.split('eps')[1].split('r')[0])
            r = float(file.split('eps')[1].split('r')[1].split('.csv')[0])
        else: 
            eps, r = 0, 0

        data = pd.read_csv(os.path.join(args.log_dir, file))

        data = np.array(data)
        validity = data[:,0]
        validity_overall = data[:, 1]

        dataset = get_dataset(file)
        if args.target_datasets != [] and dataset not in args.target_datasets:
            continue
        training_method = get_training_method(file)
        if training_method == -1:
            continue
        cfx = get_cfx(file, dataset)

        all_data.append([dataset, training_method, cfx, epochs, eps, r, validity.mean(), validity.std(),
                         validity_overall.mean(), validity_overall.std()])
        
    all_data = np.array(all_data)
    columns = ['dataset', 'training_method', 'cfx', 'epochs', 'eps', 'r', 'validity_mean', 'validity_std', 
               'validity_overall_mean', 'validity_overall_std']
    df = pd.DataFrame(all_data, columns=columns)
    df.to_csv(args.filename, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='logs/validity')
    parser.add_argument('--target_datasets', default=[], nargs='+')
    parser.add_argument('--filename', default="all_data_validity.csv", help="filename for results")
    args = parser.parse_args()
    main(args)