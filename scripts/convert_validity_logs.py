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

        data = pd.read_csv(os.path.join(args.log_dir, file))

        data = np.array(data)
        validity = data[:,0]
        validity_overall = data[:, 1]

        dataset = get_dataset(file)
        training_method = get_training_method(file)
        cfx = get_cfx(file)

        all_data.append([dataset, training_method, cfx, validity.mean(), validity.std(),
                         validity_overall.mean(), validity_overall.std()])
        
    all_data = np.array(all_data)
    columns = ['dataset', 'training_method', 'cfx', 'validity_mean', 'validity_std', 
               'validity_overall_mean', 'validity_overall_std']
    df = pd.DataFrame(all_data, columns=columns)
    df.to_csv('all_data_validity.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='logs/validity')

    args = parser.parse_args()
    main(args)