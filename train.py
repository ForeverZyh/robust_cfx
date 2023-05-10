import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

import torch.nn.functional as F

from utils import dataset
from utils import cfx
from models.IBPModel import FNN
from models.standard_model import Standard_FNN

import argparse

def train_IBP(train_data, test_data, batch_size, dim_in, num_hiddens, minmax, cfx_method):
    model = FNN(dim_in, 2, num_hiddens, epsilon=1e-2, bias_epsilon=1e-1)

    # cfx_data = dataset.Custom_Dataset("../data/german_train_lim.csv", "credit_risk")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    # NOTE: we need shuffle=False for CFX's to be in right order (or need to change how we generate CFX)
    # There should be dataset loader that returns index of the data point
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


    cfx_generator = cfx.CFX_Generator(predictor, train_data, num_layers=len(num_hiddens))
    cfx_generation_freq = 20
    eval_freq = 5
    cfx_x = None
    is_cfx = None
    for epoch in range(50):
        model.eval()
        if epoch % cfx_generation_freq == cfx_generation_freq - 1:
            # generate CFX
            if cfx_method == "proto":
                cfx_x, is_cfx = cfx_generator.run_proto(scaler=minmax)
            else:     
                cfx_x, is_cfx = cfx_generator.run_wachter(scaler=minmax)
        model.train()
        total_loss = 0
        for batch, (X, y, idx) in enumerate(train_dataloader):
            optimizer.zero_grad()
            if cfx_x is None:
                loss = model.get_loss(X, y, None, None, 0)
            else:
                this_cfx = cfx_x[idx]
                this_is_cfx = is_cfx[idx]
                loss = model.get_loss(X, y, this_cfx, this_is_cfx,
                                      0.1)  # changing lambda_ratio to 0.0 results in low CFX accuracy.
            total_loss += loss.item() * batch_size
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
    eval_freq = 10
    for epoch in range(50):
        total_loss = 0
        for batch, (X, y, _) in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss = model.get_loss(X, y)
            total_loss += loss.item() * batch_size
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
    parser.add_argument('model_name', type=str, help="filename to save the model parameters to")
    parser.add_argument('--model', type=str, default='IBP', help='IBP or standard')
    parser.add_argument('--cfx', type=str, default="wachter", help="wachter or proto")
    args = parser.parse_args()

    torch.random.manual_seed(0)
    # train_data, minmax = dataset.load_data("data/german_train.csv", "credit_risk", dataset.CREDIT_FEAT)
    # test_data, _ = dataset.load_data("data/german_test.csv", "credit_risk", dataset.CREDIT_FEAT, df_mm = minmax)
    train_data, test_data, minmax = dataset.load_data_v1("data/german_train.csv", "data/german_test.csv", "credit_risk", dataset.CREDIT_FEAT)
    

    batch_size = 64
    dim_in = train_data.num_features

    num_hiddens = [10, 10]

    if args.model == 'IBP':
        assert args.cfx in ['wachter', 'proto'], "Invalid CFX type"
        model = train_IBP(train_data, test_data, batch_size, dim_in, num_hiddens, minmax, args.cfx)
    elif args.model == 'Standard':
        model = train_standard(train_data, test_data, batch_size, dim_in, num_hiddens)
    else:
        raise ValueError('Invalid model type. Must be IBP or Standard')
    
    torch.save(model.state_dict(), 'models/' + args.model_name + '.pt')

    
