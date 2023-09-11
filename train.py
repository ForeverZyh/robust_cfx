import numpy as np
import torch
from torch.utils.data import DataLoader
import os

import torch.nn.functional as F

from utils import dataset
from utils import cfx
from models.IBPModel import FNN
from models.standard_model import Standard_FNN

import argparse


def train_IBP(train_data, test_data, batch_size, dim_in, num_hiddens, cfx_method, onehot, epsilon=None,
              bias_epsilon=None):
    model = FNN(dim_in, 2, num_hiddens, epsilon=epsilon, bias_epsilon=bias_epsilon)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    @torch.no_grad()
    def predictor(X: np.ndarray) -> np.ndarray:
        X = torch.as_tensor(X, dtype=torch.float32)
        with torch.no_grad():
            ret = model.forward_point_weights_bias(X)
            ret = F.softmax(ret, dim=1)
            ret = ret.cpu().numpy()
        return ret

    cfx_generator = cfx.CFX_Generator(predictor, train_data, num_layers=len(num_hiddens))
    cfx_generation_freq = args.cfx_generation_freq
    eval_freq = args.eval_freq
    max_epochs = args.epoch
    cfx_x = None
    is_cfx = None
    regenerate = np.ones(len(train_data)).astype(bool)
    for epoch in range(max_epochs):
        model.eval()
        if epoch % cfx_generation_freq == cfx_generation_freq - 1 and (epoch != max_epochs - 1):

            # generate CFX
            # TODO parallelize CFX generation? might not be necessary if moving to GPUs
            if not args.inc_regenerate:
                regenerate = np.ones(len(train_data)).astype(bool)
            if cfx_method == "proto":
                cfx_x, is_cfx = cfx_generator.run_proto(scaler=None, theta=args.proto_theta, onehot=onehot,
                                                        CEs=cfx_x, is_CE=is_cfx, regenerate=regenerate)
            else:
                cfx_x, is_cfx = cfx_generator.run_wachter(scaler=None, max_iter=args.wachter_max_iter,
                                                          CEs=cfx_x, is_CE=is_cfx, regenerate=regenerate,
                                                          lam_init=args.wachter_lam_init,
                                                          max_lam_steps=args.wachter_max_lam_steps)
        model.train()
        total_loss = 0
        for batch, (X, y, idx) in enumerate(train_dataloader):
            optimizer.zero_grad()
            if cfx_x is None:
                loss = model.get_loss(X, y, None, None, 0)
            else:
                this_cfx = cfx_x[idx]
                this_is_cfx = is_cfx[idx]
                loss = model.get_loss(X, y, this_cfx, this_is_cfx, args.ratio)
            total_loss += loss.item() * X.shape[0]
            loss.backward()
            optimizer.step()

        if epoch % eval_freq == 0:
            print("Epoch", str(epoch), "loss:", total_loss / len(train_data))
            model.eval()
            acc_cnt = 0
            with torch.no_grad():
                for X, y, _ in test_dataloader:
                    y_pred = model.forward_point_weights_bias(X.float()).argmax(dim=-1)
                    acc_cnt += torch.sum(y_pred == y).item()
            print("Epoch", str(epoch), "accuracy:", acc_cnt / len(test_data))
        model.eval()
        with torch.no_grad():
            # make sure CFX is valid
            pre_valid = 0
            post_valid = 0
            if cfx_x is not None:
                for batch_id, (X, y, idx) in enumerate(train_dataloader):
                    cfx_y = model.forward_point_weights_bias(cfx_x[idx]).argmax(dim=-1)
                    pre_valid += torch.sum(is_cfx[idx]).item()
                    is_cfx[idx] = is_cfx[idx] & (cfx_y == (1 - y)).bool()
                    post_valid += torch.sum(is_cfx[idx]).item()
                print("Epoch", str(epoch), "CFX valid:", pre_valid, post_valid)
                regenerate = np.logical_not(is_cfx)

    model.eval()
    with torch.no_grad():
        # eval on train for right now since we have those CFX
        total_samples, correct = 0, 0
        for X, y, _ in train_dataloader:
            total_samples += len(X)
            X = X.float()
            y_pred = model.forward_point_weights_bias(X).argmax(dim=-1)
            correct += torch.sum((y_pred == y).float()).item()
            # If we want CFX accuracy, need to re-run CFX generation
            # cfx_y_pred = model.forward_point_weights_bias(cfx_data.X).argmax(dim=-1)
            # print("Train CFX Acc:", torch.mean((cfx_y_pred == (1 - y)).float()).item())
        print("Train accuracy: ", round(correct / total_samples, 4))
        total_samples, correct = 0, 0
        for X, y, _ in test_dataloader:
            total_samples += len(X)
            X = X.float()
            y_pred = model.forward_point_weights_bias(X).argmax(dim=-1)
            correct += torch.sum((y_pred == y).float()).item()
        print("Test accuracy: ", round(correct / total_samples, 4))

    return model


def train_standard(train_data, test_data, batch_size, dim_in, num_hiddens):
    model = Standard_FNN(dim_in, 2, num_hiddens)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    @torch.no_grad()
    def predictor(X: np.ndarray) -> np.ndarray:
        X = torch.as_tensor(X, dtype=torch.float32)
        with torch.no_grad():
            ret = model.forward_point_weights_bias(X)
            ret = F.softmax(ret, dim=1)
            ret = ret.cpu().numpy()
            # print(ret)
        return ret

    model.train()
    eval_freq = args.eval_freq
    for epoch in range(args.epoch):
        total_loss = 0
        for batch, (X, y, _) in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss = model.get_loss(X, y)
            total_loss += loss.item() * X.shape[0]
            loss.backward()
            optimizer.step()

        if epoch % eval_freq == 0:
            print("Epoch", str(epoch), "loss:", total_loss / len(train_data))

    model.eval()
    with torch.no_grad():
        total_samples, correct = 0, 0
        for X, y, _ in train_dataloader:
            total_samples += len(X)
            X = X.float()
            y_pred = model.forward(X).argmax(dim=-1)
            correct += torch.sum((y_pred == y).float()).item()
        print("Train accuracy: ", round(correct / total_samples, 4))
        total_samples, correct = 0, 0
        for X, y, _ in test_dataloader:
            total_samples += len(X)
            X = X.float()
            y_pred = model.forward(X).argmax(dim=-1)
            correct += torch.sum((y_pred == y).float()).item()
        print("Test accuracy: ", round(correct / total_samples, 4))
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str,
                        help="filename to save the model parameters to (don't include .pt )")
    parser.add_argument('--save_dir', type=str, default="trained_models", help="directory to save models to")
    parser.add_argument('--model', type=str, default='IBP', help='IBP or Standard', choices=['IBP', 'Standard'])

    # cfx args
    parser.add_argument('--cfx', type=str, default="wachter", help="wachter or proto", choices=["wachter", "proto"])
    parser.add_argument('--onehot', action='store_true', help='whether to use one-hot encoding')
    parser.add_argument('--proto_theta', type=float, default=100, help='theta for proto')
    parser.add_argument('--wachter_max_iter', type=int, default=100, help='max iter for wachter')
    parser.add_argument('--wachter_lam_init', type=float, default=1e-3, help='initial lambda for wachter')
    parser.add_argument('--wachter_max_lam_steps', type=int, default=10, help='max lambda steps for wachter')

    # training args
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--eval_freq', type=int, default=10, help='frequency of evaluation')

    # IBP training args
    parser.add_argument('--cfx_generation_freq', type=int, default=20, help='frequency of CFX generation')
    parser.add_argument('--ratio', type=float, default=0.1, help='ratio of CFX loss')
    parser.add_argument('--inc_regenerate', action='store_true',
                        help='whether to regenerate CFXs incrementally for those that are no longer CFX each time')
    parser.add_argument('--epsilon', type=float, default=1e-2, help='epsilon for IBP')
    parser.add_argument('--bias_epsilon', type=float, default=1e-3, help='bias epsilon for IBP')
    args = parser.parse_args()

    torch.random.manual_seed(0)

    if args.cfx == 'proto':
        feature_types = dataset.CREDIT_FEAT_PROTO
    else:
        feature_types = dataset.CREDIT_FEAT
    if args.onehot:
        train_data, min_vals, max_vals = dataset.load_data("data/german_train.csv", "credit_risk", feature_types)
        test_data, _, _ = dataset.load_data("data/german_test.csv", "credit_risk", feature_types, min_vals, max_vals)
    else:
        train_data, test_data, minmax = dataset.load_data_v1("data/german_train.csv", "data/german_test.csv",
                                                             "credit_risk",
                                                             feature_types)
    batch_size = 64
    dim_in = train_data.num_features
    num_hiddens = [10, 10]

    if args.model == 'IBP':
        model = train_IBP(train_data, test_data, batch_size, dim_in, num_hiddens, args.cfx, args.onehot,
                          epsilon=args.epsilon, bias_epsilon=args.bias_epsilon)
    elif args.model == 'Standard':
        model = train_standard(train_data, test_data, batch_size, dim_in, num_hiddens)
    else:
        raise ValueError('Invalid model type. Must be IBP or Standard')

    torch.save(model.state_dict(), os.path.join(args.save_dir, args.model_name + '.pt'))
