import argparse
import json
import os

import torch.nn as nn
# import onnx
# import tf2onnx
import tensorflow as tf
from torch.autograd import Variable
import torch

from train import prepare_data_and_model
import models.IBPModel as IBPModel
import models.IBPModel_tf as IBPModel_tf

from utils.utilities import seed_everything, FNNDims


def main(args):
    ret = prepare_data_and_model(args)
    train_data, test_data, model_pytorch, minmax = ret["train_data"], ret["test_data"], ret["model"], ret[
        "preprocessor"]
    model_pytorch.load(os.path.join(args.model_dir, args.model))

    if args.config["act"] == 0:
        act = tf.keras.activations.relu
    elif args.config["act"] > 0:
        act = tf.keras.layers.LeakyReLU(args.config["act"])
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

    model.save(os.path.join(args.save_dir, args.model[:-3]))
    model.load(os.path.join(args.save_dir, args.model[:-3]))
    print(model.model.predict(train_data.X[:10]), model_pytorch.predict_proba(train_data.X[:10]))
    print(train_data.y[:10])

    # convert model from pytorch to tensorflow going layer-by-layer
    # structure of model is in models.IBPModel.py class Counternet
    # has two submodels, one for the predictor and one for the explainer
    # The predictor is an EncDec with the structure 
    #         self.encoder = MultilayerPerception([enc_dims.in_dim] + enc_dims.hidden_dims, epsilon_ratio, activation,
    #                                     dropout=dropout)
    # self.decoder = MultilayerPerception([dec_dims.in_dim] + dec_dims.hidden_dims, epsilon_ratio, activation,
    #                                     dropout=dropout)
    # self.final_fc = BoundedLinear(dec_dims.hidden_dims[-1], num_outputs, epsilon_ratio)
    # first, convert the predictor (EncDec) architecture
    # keras_encdec.encoder = keras.Sequential()
    # # create first multilayer perceptron ([enc_dims.in_dim] + enc_dims.hidden_dims, epsilon_ratio = args.config["eps_ratio"], act, dropout, dropout=args.config["dropout"])
    # mult_perc = keras.layers.Dense(enc_dims.hidden_dims[0], input_shape=(enc_dims.in_dim,), activation=act)
    # keras_encdec.encoder.add(mult_perc)
    # # add remaining hidden layers for multilayer perceptron
    # for i in range(1, len(enc_dims.hidden_dims)):
    #     mult_perc = keras.layers.Dense(enc_dims.hidden_dims[i], activation=act)
    #     keras_encdec.encoder.add(mult_perc)
    # # add final layer for multilayer perceptron
    # final_layer = keras.encoder.layers.Dense(enc_dims.hidden_dims[-1], activation=act)

    # didn't work
    # input_var = Variable(torch.tensor(train_data.X[:1]))
    # torch.onnx.export(model, input_var, "pytorch_model.onnx", verbose=True, input_names=["input"], output_names=["output"])

    # onnx_model = onnx.load("pytorch_model.onnx")
    # tf_rep = tf2onnx.convert.from_path(onnx_model, input_signature=[("input", tf.TensorType(shape=[None, 10], dtype=tf.float32))], opset=13)
    # graph_def = tf_rep.graph.as_graph_def() 

    # tf.compat.v1.reset_default_graph()
    # tf_model = tf.compat.v1.GraphDef()
    # tf.import_graph_def(graph_def, name="")

    # input_node = tf_model.get_tensor_by_name("input:0")
    # output_node = tf_model.get_tensor_by_name("output:0")

    # with tf.compat.v1.Session() as sess:
    #     tf_output = sess.run(output_node, feed_dict={input_node: input_var.numpy()})

    # # save the model 
    # tf_model.save("tf_model.pb")

    # DIDN'T WORK - cannot install all dependencies for pytorch2keras
    # model.load(os.path.join(args.model_dir, args.model))

    # input_var = Variable(train_data.X[:1])
    # shape = train_data.X.shape[1:]
    # k_model = pytorch_to_keras(model, input_var, [shape], verbose=True, names='short')  

    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)
    # k_model.save(os.path.join(args.save_dir, args.model[:-3] + '.h5'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="name of model to convert")
    parser.add_argument("dataset")
    parser.add_argument("--model_dir", default="trained_models", help="directory where pytorch model is saved")
    parser.add_argument("--save_dir", default="sns/saved_keras_models", help="directory to save keras model to")

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
