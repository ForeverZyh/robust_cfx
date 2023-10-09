import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import os
import json
import argparse

import torch.nn.functional as F
import torch.nn as nn
import wandb

from utils import dataset
from utils import cfx
from models.IBPModel import FNN, VerifyModel, CounterNet
from utils.utilities import seed_everything, FNNDims


def eval_chunk(model, test_dataloader, val_dataloader, train_dataloader, epoch, cfx_x, is_cfx):
    model.eval()
    dataloaders = [val_dataloader, train_dataloader, test_dataloader]
    tasks = ["validation", "train", "test"]
    update = {}
    with torch.no_grad():
        for dataloader, task in zip(dataloaders, tasks):
            total_loss, acc_cnt, total_samples = 0, 0, 0
            for X, y, idx in dataloader:
                if cfx_x is None:
                    loss = model.get_loss(X, y, None, None, args.ratio, args.tightness)
                else:
                    this_cfx = cfx_x[idx]
                    this_is_cfx = is_cfx[idx]
                    loss = model.get_loss(X, y, this_cfx, this_is_cfx, args.ratio, args.tightness)
                total_loss += loss.item() * len(X)
                total_samples += len(X)
                y_pred = model.forward_point_weights_bias(X.float()).argmax(dim=-1)
                acc_cnt += torch.sum(y_pred == y).item()

            print("Epoch", str(epoch), f"{task} acc: ", acc_cnt / total_samples, "test loss: ",
                  total_loss / total_samples)
            update[f"{task}_acc"] = acc_cnt / total_samples
            update[f"{task}_loss"] = total_loss / total_samples
    return update


def eval_chunk_counternet(model, val_dataloader, epoch):
    model.eval()
    acc_cnt = 0
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for X, y, _ in val_dataloader:
            cfx_new, _ = model.forward(X, hard=True)
            is_cfx_new = model.forward_point_weights_bias(cfx_new).argmax(dim=1) == 1 - y
            y_pred = model.forward_point_weights_bias(X.float()).argmax(dim=-1)
            acc_cnt += torch.sum(y_pred == y).item()
            total_loss += model.get_loss(X, y, cfx_new, is_cfx_new, args.ratio, args.tightness).item() * len(X)
            total_samples += len(X)

    print("Epoch", str(epoch), "Test accuracy:", acc_cnt / total_samples, "Test loss:", total_loss / total_samples)
    return {"test_acc": acc_cnt / total_samples, "test_loss": total_loss / total_samples}


def eval_train_test_chunk(model, train_dataloader, test_dataloader):
    model.eval()
    dataloaders = [train_dataloader, test_dataloader]
    tasks = ["train_acc_final", "test_acc_final"]
    with torch.no_grad():
        for dataloader, task in zip(dataloaders, tasks):
            total_samples, correct = 0, 0
            for X, y, _ in dataloader:
                total_samples += len(X)
                X = X.float()
                y_pred = model.forward_point_weights_bias(X).argmax(dim=-1)
                correct += torch.sum((y_pred == y).float()).item()
            print(f"{task}: ", round(correct / total_samples, 4))
            if args.wandb is not None:
                args.wandb.summary[task] = round(correct / total_samples, 4)


def eval_cfx_chunk(model, cfx_dataloader, cfx_x, is_cfx, epoch):
    model.eval()
    with torch.no_grad():
        # make sure CFX is valid
        post_valid = 0
        for (X, y, idx) in cfx_dataloader:
            cfx_y = model.forward_point_weights_bias(cfx_x[idx]).argmax(dim=-1)
            is_cfx[idx] = is_cfx[idx] & (cfx_y == (1 - y)).bool()
            post_valid += torch.sum(is_cfx[idx]).item()

        if args.wandb is None:
            print("Epoch", str(epoch), "CFX valid:", post_valid)
        return {"valid_cfx": post_valid}


def train_IBP(train_data, test_data, model: VerifyModel, cfx_method, onehot, filename):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.config["weight_decay"])
    cfx_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)  # for CFX test

    @torch.no_grad()
    def predictor(X: np.ndarray) -> np.ndarray:
        X = torch.as_tensor(X, dtype=torch.float32)
        with torch.no_grad():
            ret = model.forward_point_weights_bias(X)
            ret = F.softmax(ret, dim=1)
            ret = ret.cpu().numpy()
        return ret

    # not sure why we need num_layers here.
    cfx_generator = cfx.CFX_Generator(predictor, train_data, num_layers=None)
    ori_train_len = len(train_data)

    val_size = int(ori_train_len // 8)  # 1/8 of the training set is used for validation
    train_data, val_data = random_split(train_data, [ori_train_len - val_size, val_size])
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    cfx_generation_freq = args.cfx_generation_freq
    max_epochs = args.epoch
    cfx_x = None
    is_cfx = None
    regenerate = np.ones(ori_train_len).astype(bool)
    best_val_loss = np.inf
    best_epoch = -1
    for epoch in range(max_epochs):
        model.eval()
        wandb_log = {}
        if epoch % cfx_generation_freq == 0 and epoch > 0:
            # generate CFX
            # TODO parallelize CFX generation? might not be necessary if moving to GPUs
            if not args.inc_regenerate:
                regenerate = np.ones(ori_train_len).astype(bool)
            if cfx_method == "proto":
                cfx_x, is_cfx = cfx_generator.run_proto(scaler=None, theta=args.proto_theta, onehot=onehot,
                                                        CEs=cfx_x, is_CE=is_cfx, regenerate=regenerate)
            else:
                cfx_x, is_cfx = cfx_generator.run_wachter(scaler=None, max_iter=args.wachter_max_iter,
                                                          CEs=cfx_x, is_CE=is_cfx, regenerate=regenerate,
                                                          lam_init=args.wachter_lam_init,
                                                          max_lam_steps=args.wachter_max_lam_steps)

        if cfx_x is not None:
            wandb_log.update(eval_cfx_chunk(model, cfx_dataloader, cfx_x, is_cfx, epoch))

        model.train()
        total_loss = 0
        for batch, (X, y, idx) in enumerate(train_dataloader):
            optimizer.zero_grad()
            if cfx_x is None:
                loss = model.get_loss(X, y, None, None, args.ratio, args.tightness)
            else:
                this_cfx = cfx_x[idx]
                this_is_cfx = is_cfx[idx]
                loss = model.get_loss(X, y, this_cfx, this_is_cfx, args.ratio, args.tightness)
            total_loss += loss.item() * X.shape[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.config["clip_grad_norm"])
            optimizer.step()

        # wandb_log.update({"train_loss": total_loss / len(train_data)})
        if args.wandb is None:
            print("Epoch", str(epoch), "train_loss:", total_loss / len(train_data))

        wandb_log.update(eval_chunk(model, test_dataloader, val_dataloader, train_dataloader, epoch, cfx_x, is_cfx))
        if best_val_loss > wandb_log["test_loss"]:
            best_val_loss = wandb_log["test_loss"]
            best_epoch = epoch
            model.save(filename)

        if args.wandb is not None:
            args.wandb.log(wandb_log, commit=True)

    model.load(filename)
    eval_train_test_chunk(model, train_dataloader, test_dataloader)
    if args.wandb is not None:
        args.wandb.summary["best_epoch"] = best_epoch
    print("best epoch: ", best_epoch)
    return model


def train_IBP_counternet(train_data, test_data, model: CounterNet, filename):
    optimizer_1 = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.config["weight_decay"])
    optimizer_2 = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.config["weight_decay"])
    cfx_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)  # for CFX test
    ori_train_len = len(train_data)
    val_size = int(ori_train_len // 8)  # 1/8 of the training set is used for validation
    train_data, val_data = random_split(train_data, [ori_train_len - val_size, val_size])
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    cfx_generation_freq = args.cfx_generation_freq
    max_epochs = args.epoch
    cfx_x = None
    is_cfx = None
    best_val_loss = np.inf
    best_epoch = -1
    for epoch in range(max_epochs):
        wandb_log = {}
        model.eval()
        if epoch % cfx_generation_freq == 0 and epoch > 0:
            if not args.inc_regenerate or is_cfx is None:
                regenerate = torch.ones(ori_train_len).bool()
            else:
                regenerate = ~is_cfx
            with torch.no_grad():
                cfx_new_list = []
                is_cfx_new_list = []
                for X, y, _ in cfx_dataloader:
                    cfx_new, _ = model.forward(X, hard=True)
                    is_cfx_new = model.forward_point_weights_bias(cfx_new).argmax(dim=1) == 1 - y
                    cfx_new_list.append(cfx_new)
                    is_cfx_new_list.append(is_cfx_new)

            if cfx_x is None:
                cfx_x = torch.cat(cfx_new_list, dim=0)
                is_cfx = torch.cat(is_cfx_new_list, dim=0)
            else:
                cfx_x = torch.where(regenerate.unsqueeze(-1), torch.cat(cfx_new_list, dim=0), cfx_x)
                is_cfx = torch.where(regenerate, torch.cat(is_cfx_new_list, dim=0), is_cfx)

        if cfx_x is not None:
            wandb_log.update(eval_cfx_chunk(model, cfx_dataloader, cfx_x, is_cfx, epoch))

        model.train()
        predictor_loss = 0
        explainer_loss = 0
        for batch, (X, y, idx) in enumerate(train_dataloader):
            # predictor step
            optimizer_1.zero_grad()
            if cfx_x is None:
                loss = model.get_predictor_loss(X, y, None, None, args.ratio, args.tightness)
            else:
                this_cfx = cfx_x[idx]
                this_is_cfx = is_cfx[idx]
                loss = model.get_predictor_loss(X, y, this_cfx, this_is_cfx, args.ratio, args.tightness)
            predictor_loss += loss.item() * X.shape[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.config["clip_grad_norm"])
            optimizer_1.step()

        for batch, (X, y, idx) in enumerate(train_dataloader):
            # explainer step
            optimizer_2.zero_grad()
            loss = model.get_explainer_loss(X, y)
            explainer_loss += loss.item() * X.shape[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.config["clip_grad_norm"])
            optimizer_2.step()

        wandb_log.update({"predictor_loss": predictor_loss / len(train_data),
                          "explainer_loss": explainer_loss / len(train_data)})
        if args.wandb is None:
            print("predictor_loss: ", predictor_loss / len(train_data))
            print("explainer_loss: ", explainer_loss / len(train_data))

        wandb_log.update(eval_chunk_counternet(model, val_dataloader, epoch))
        if best_val_loss > wandb_log["test_loss"]:
            best_val_loss = wandb_log["test_loss"]
            model.save(filename)
            best_epoch = epoch

        if args.wandb is not None:
            args.wandb.log(wandb_log, commit=True)

    model.load(filename)
    eval_train_test_chunk(model, train_dataloader, test_dataloader)
    if args.wandb is not None:
        args.wandb.summary["best_epoch"] = best_epoch
    print("best epoch: ", best_epoch)
    return model


def prepare_data_and_model(args):
    ret = {"preprocessor": None, "train_data": None, "test_data": None, "model": None, "minmax": None}
    if args.config["dataset_name"] == "german_credit":
        if args.cfx == 'proto':
            feature_types = dataset.CREDIT_FEAT_PROTO
        else:
            feature_types = dataset.CREDIT_FEAT
        if args.onehot:
            train_data, preprocessor = dataset.load_data("data/german_train.csv", "credit_risk", feature_types)
            test_data, _, = dataset.load_data("data/german_test.csv", "credit_risk", feature_types, preprocessor)
            ret["preprocessor"] = preprocessor
        else:
            train_data, test_data, minmax = dataset.load_data_v1("data/german_train.csv", "data/german_test.csv",
                                                                 "credit_risk", feature_types)
            ret["minmax"] = minmax
    elif args.config["dataset_name"] == "heloc":
        feature_types = dataset.HELOC_FEAT
        train_data, preprocessor = dataset.load_data("data/heloc_train.csv", "label", feature_types)
        test_data, _, = dataset.load_data("data/heloc_test.csv", "label", feature_types, preprocessor)
        ret["preprocessor"] = preprocessor
    elif args.config["dataset_name"] == "ctg":
        feature_types = dataset.CTG_FEAT
        train_data, preprocessor = dataset.load_data("data/ctg_train.csv", "label", feature_types)
        test_data, _, = dataset.load_data("data/ctg_test.csv", "label", feature_types, preprocessor)
        ret['preprocessor'] = preprocessor
    else:
        raise NotImplementedError(f"Dataset {args.config['dataset_name']} not implemented")
    ret["train_data"] = train_data
    ret["test_data"] = test_data
    args.batch_size = args.config["batch_size"]
    dim_in = train_data.num_features
    if args.config["act"] == 0:
        act = nn.ReLU
    elif args.config["act"] > 0:
        act = lambda: nn.LeakyReLU(args.config["act"])
    else:
        raise NotImplementedError("Activation function not implemented")

    if args.cfx == "counternet":
        if args.config["dataset_name"] == "german_credit":
            assert args.onehot, "Counternet should work with onehot"
        enc_dims = FNNDims(dim_in, args.config["encoder_dims"])
        pred_dims = FNNDims(None, args.config["decoder_dims"])
        exp_dims = FNNDims(None, args.config["explainer_dims"])
        model = CounterNet(enc_dims, pred_dims, exp_dims, 2, epsilon=args.epsilon, bias_epsilon=args.bias_epsilon,
                           activation=act, dropout=args.config["dropout"], preprocessor=preprocessor,
                           config=args.config)
    else:
        model_ori = FNN(dim_in, 2, args.config["FNN_dims"], epsilon=args.epsilon, bias_epsilon=args.bias_epsilon,
                        activation=act)
        model = VerifyModel(model_ori, dummy_input_shape=train_data.X[:2].shape, loss_func=args.config["loss_1"])
    ret["model"] = model
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str,
                        help="filename to save the model parameters to (don't include .pt )")
    parser.add_argument('--save_dir', type=str, default="trained_models", help="directory to save models to")
    parser.add_argument('--model', type=str, default=None, help='IBP or Standard', choices=['IBP', 'Standard'])
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--config', type=str, default="assets/german_credit.json",
                        help='config file for the dataset and the model')
    parser.add_argument('--wandb', action='store_true', help='whether to log to wandb')

    # cfx args
    parser.add_argument('--cfx', type=str, default="wachter", help="wachter or proto",
                        choices=["wachter", "proto", "counternet"])
    parser.add_argument('--onehot', action='store_true', help='whether to use one-hot encoding')
    parser.add_argument('--proto_theta', type=float, default=100, help='theta for proto')
    parser.add_argument('--wachter_max_iter', type=int, default=100, help='max iter for wachter')
    parser.add_argument('--wachter_lam_init', type=float, default=1e-3, help='initial lambda for wachter')
    parser.add_argument('--wachter_max_lam_steps', type=int, default=10, help='max lambda steps for wachter')

    # training args
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs to train')
    # lr has been moved to the config file
    # parser.add_argument('--lr', type=float, default=0.01, help='learning rate')

    # IBP training args
    parser.add_argument('--cfx_generation_freq', type=int, default=20, help='frequency of CFX generation')
    parser.add_argument('--ratio', type=float, default=0.1, help='ratio of CFX loss')
    parser.add_argument('--tightness', choices=["ours", "ibp", "crownibp", "none"], default="ours",
                        help='the tightness of the bound. None means not using the bound.')
    parser.add_argument('--inc_regenerate', action='store_true',
                        help='whether to regenerate CFXs incrementally for those that are no longer CFX each time')
    parser.add_argument('--epsilon', type=float, default=1e-2, help='epsilon for IBP')
    parser.add_argument('--bias_epsilon', type=float, default=1e-3, help='bias epsilon for IBP')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        args.config = json.load(f)
    args.lr = args.config["lr"]
    if args.wandb:
        args.wandb = wandb.init(project="robust_cfx", name=args.model_name, config=args.__dict__)
    else:
        args.wandb = None

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    seed_everything(args.seed)
    if args.model == "Standard":
        args.ratio = 0
        args.cfx_generation_freq = args.epoch + 1  # never generate cfx
        args.tightness = "none"

    ret = prepare_data_and_model(args)

    if args.cfx == "counternet":
        model = train_IBP_counternet(ret["train_data"], ret["test_data"], ret["model"],
                                     os.path.join(args.save_dir, args.model_name))
    else:
        model = train_IBP(ret["train_data"], ret["test_data"], ret["model"], args.cfx, args.onehot,
                          os.path.join(args.save_dir, args.model_name))

    if args.wandb is not None:
        args.wandb.finish()
