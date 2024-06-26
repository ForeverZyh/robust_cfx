import tensorflow as tf
import numpy as np

from consistency import IterativeSearch
from consistency import PGDsL2
from consistency import StableNeighborSearch

import os
import argparse
import pickle
import json

import models.IBPModel_tf as IBPModel_tf
from utils.utilities import FNNDims
from utils.dataset import prepare_data

def main(args):
    ret = prepare_data(args)
    X_train = np.array(ret["train_data"].X).astype(np.float32)
    X_test = np.array(ret["test_data"].X).astype(np.float32)

    # set seed 
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

    # next, need to get model
    model = IBPModel_tf.CounterNet(enc_dims, pred_dims, exp_dims, 2,
                                   epsilon_ratio=args.config["eps_ratio"],
                                   activation=act, dropout=args.config["dropout"], preprocessor=None,
                                   config=args.config)

    model.build()
    model.load(os.path.join(args.save_dir, args.model))

    model = model.model

    original_preds = model.predict(X_test[:args.num_to_run]).argmax(axis=-1)

    sns_fn = StableNeighborSearch(model,
                                  clamp=[X_train.min(), X_train.max()],
                                  num_classes=2,
                                  sns_eps=0.1,
                                  sns_nb_iters=100,
                                  sns_eps_iter=1.e-3,
                                  n_interpolations=20)

    if args.technique == 'l1':
        L1_iter_search = IterativeSearch(model,
                                         clamp=[X_train.min(), X_train.max()],
                                         num_classes=2,
                                         eps=0.3,
                                         nb_iters=40,
                                         eps_iter=0.01,
                                         norm=1,
                                         sns_fn=sns_fn)

        cf, pred_cf, is_valid = L1_iter_search(X_test[:args.num_to_run])

    elif args.technique == 'l2':
        L2_iter_search = IterativeSearch(model,
                                         clamp=[X_train.min(), X_train.max()],
                                         num_classes=2,
                                         eps=0.3,
                                         nb_iters=40,
                                         eps_iter=0.01,
                                         norm=2,
                                         sns_fn=sns_fn)
        cf, pred_cf, is_valid = L2_iter_search(X_test[:128])
    elif args.technique == 'pgd':
        pgd_iter_search = PGDsL2(model,
                                 clamp=[X_train.min(), X_train.max()],
                                 num_classes=2,
                                 eps=2.0,
                                 nb_iters=100,
                                 eps_iter=0.04,
                                 sns_fn=sns_fn)
        cf, pred_cf, is_valid = pgd_iter_search(X_test[:args.num_to_run], num_interpolations=10, batch_size=64)

    if not ( is_valid.sum() / len(is_valid) ) == 1:
        # their validity is just true/false -- unsure what exactly.
        print("Problem: is_valid is ", is_valid.sum()/len(is_valid))

    # check validity by seeing that pred_cf != original preds
    validity = (is_valid & (original_preds != pred_cf)).astype(int)
    # save cfx
    cfx_filename = os.path.join(args.cfx_save_dir, args.model + "_sns.npy")
    with open(cfx_filename, 'wb') as f:
        pickle.dump((cf, validity), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('dataset')
    parser.add_argument('--save_dir', default="sns/saved_keras_models")
    parser.add_argument('--technique', default="l1", choices=['l1', 'l2', 'pgd'],
                        help="how to generate CFX during SNS (l1, l2, pgd)")
    parser.add_argument('--cfx_save_dir', default="sns/saved_cfxs", help="where to save generated cfxs")
    parser.add_argument('--seed', type=int, default=0, help='random seed')    
    parser.add_argument('--num_to_run', type=int, default=None)

    args = parser.parse_args()

    args.finetune=False
    args.model_type = args.model.split(args.dataset)[0]

    if args.num_to_run is None:
        if args.dataset == 'ctg':
            args.num_to_run = 500
        else:
            args.num_to_run = 1000

    if not os.path.exists(args.cfx_save_dir):
        os.makedirs(args.cfx_save_dir)

    args.config = f'assets/{args.dataset}.json'
    with open(args.config, 'r') as f:
        args.config = json.load(f)

    args.remove_pct = None
    main(args)
