import tensorflow as tf 
import numpy as np

from consistency import IterativeSearch
from consistency import PGDsL2
from consistency import StableNeighborSearch

from utils import load_dataset
from utils import invalidation

import os 
import argparse
import pickle


# REMAINING TODO
# 1. change model from pytorch to tensorflow
#       trying pytorch2keras --> if it doesn't work, go layer-by-layer and copy weights
# 2. √ change how CFXs are saved here
# 3. √ add additional datasets to the local dataset directory and util file

def main(args):
    (X_train, y_train), (X_test, y_test), n_classes = load_dataset('TaiwaneseCredit', path_to_data_dir='dataset/data')

    # set seed 
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # next, need to get model
    baseline_model = 0

    original_preds = baseline_model.predict(X_test[:128]).argmax(axis=-1)

    sns_fn = StableNeighborSearch(baseline_model,
                clamp=[X_train.min(), X_train.max()],
                num_classes=2,
                sns_eps=0.1,
                sns_nb_iters=100,
                sns_eps_iter=1.e-3,
                n_interpolations=20)
    
    if args.technique == 'l1':
        L1_iter_search = IterativeSearch(baseline_model,
                                        clamp=[X_train.min(), X_train.max()],
                                        num_classes=2,
                                        eps=0.3,
                                        nb_iters=40,
                                        eps_iter=0.01,
                                        norm=1,
                                        sns_fn=sns_fn)
                                        
        cf, pred_cf, is_valid = L1_iter_search(X_test[:128])
    elif args.technique == 'l2':
        L2_iter_search = IterativeSearch(baseline_model,
                                clamp=[X_train.min(), X_train.max()],
                                num_classes=2,
                                eps=0.3,
                                nb_iters=40,
                                eps_iter=0.01,
                                norm=2,
                                sns_fn=sns_fn)
        cf, pred_cf, is_valid = L2_iter_search(X_test[:128])
    elif args.technique == 'pgd':
        pgd_iter_search = PGDsL2(baseline_model,
                        clamp=[X_train.min(), X_train.max()],
                        num_classes=2,
                        eps=2.0,
                        nb_iters=100,
                        eps_iter=0.04,
                        sns_fn=sns_fn)
        cf, pred_cf, is_valid = pgd_iter_search(X_test[:128], num_interpolations=10, batch_size=64)


    if not is_valid:
        # their validity is just true/false -- unsure what exactly.
        print("Problem: is_valid is ",is_valid)

    # check validity by seeing that pred_cf != original preds
    validity = (original_preds != pred_cf).astype(int)

    # save cf
    if not os.path.exists(args.cfx_save_dir):
        os.makedirs(args.cfx_save_dir)

    cfx_filename = os.path.join(args.cfx_save_dir, args.dataset + "_" + args.technique + "_sns" + str(args.seed) + ".npy")
    with open(cfx_filename, 'wb') as f:
        pickle.dump((cf, validity), f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--model_dir', default="trained_models")
    parser.add_argument('--technique', default="l1", choices=['l1', 'l2', 'pgd'], 
                        help="how to generate CFX during SNS (l1, l2, pgd)")
    parser.add_argument('--cfx_save_dir', default="sns/saved_cfxs", help="where to save generated cfxs")
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    args = parser.parse_args()

    main(args)