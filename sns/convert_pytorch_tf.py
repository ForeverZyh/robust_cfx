import argparse
import json
import os

import tensorflow as tf

from train import prepare_data_and_model
import models.IBPModel_tf as IBPModel_tf

from utils.utilities import FNNDims


def main(args):
    ret = prepare_data_and_model(args)
    train_data, test_data, model_pytorch, minmax = ret["train_data"], ret["test_data"], ret["model"], ret[
        "preprocessor"]
    
    model_pytorch.load(os.path.join(args.save_dir, args.model_name))

    if args.config["act"] == 0:
        act = tf.keras.activations.relu
    elif args.config["act"] > 0:
        act = lambda: tf.keras.layers.LeakyReLU(args.config["act"])
    dim_in = train_data.num_features_processed
    enc_dims = FNNDims(dim_in, args.config["encoder_dims"])
    pred_dims = FNNDims(None, args.config["decoder_dims"])
    exp_dims = FNNDims(None, args.config["explainer_dims"])
    # create keras model
    model = IBPModel_tf.CounterNet(enc_dims, pred_dims, exp_dims, 2,
                                   epsilon_ratio=args.config["eps_ratio"],
                                   activation=act, dropout=args.config["dropout"], preprocessor=minmax,
                                   config=args.config)
    model.build()
    # copy weights from model_pytorch to model
    # first, copy weights from model_pytorch.encoder_net_ori.encoder. This is a MultiLayerPerceptron
    for i, b in enumerate(model_pytorch.encoder_net_ori.encoder.blocks):
        model.encoder_net_ori.encoder.blocks[i * 2].set_weights(  # skip the activation layers
            [b.linear.linear.weight.transpose(0, 1).detach().numpy(), b.linear.linear.bias.detach().numpy()])

    for i, b in enumerate(model_pytorch.encoder_net_ori.decoder.blocks):
        model.encoder_net_ori.decoder.blocks[i * 2].set_weights(
            [b.linear.linear.weight.transpose(0, 1).detach().numpy(), b.linear.linear.bias.detach().numpy()])

    model.encoder_net_ori.final_fc.set_weights(
        [model_pytorch.encoder_net_ori.final_fc.linear.weight.transpose(0, 1).detach().numpy(),
         model_pytorch.encoder_net_ori.final_fc.linear.bias.detach().numpy()])

    print(model.model.predict(train_data.X[:10]))
    model.save(os.path.join(args.new_model_dir, args.model_name))
    model.load(os.path.join(args.new_model_dir, args.model_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="name of model to convert")
    parser.add_argument("dataset")
    parser.add_argument("--save_dir", default="trained_models", help="directory where pytorch model is saved")
    parser.add_argument("--new_model_dir", default="sns/saved_keras_models", help="directory to save keras model to")

    args = parser.parse_args()

    if args.dataset == 'german':
        args.config = 'assets/german_credit.json'
    else:
        args.config = 'assets/' + args.dataset + '.json'

    with open(args.config, 'r') as f:
        args.config = json.load(f)

    args.remove_pct = None
    args.cfx = 'counternet'
    args.onehot = True
    main(args)
