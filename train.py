import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

import torch.nn.functional as F

from utils import dataset
from utils import cfx
from models.IBPModel import FNN

if __name__ == '__main__':
    torch.random.manual_seed(0)
    train_data = dataset.Custom_Dataset("data/german_train.csv", "credit_risk")
    test_data = dataset.Custom_Dataset("data/german_test.csv", "credit_risk")
    minmax = MinMaxScaler(clip=True)
    train_data.X = minmax.fit_transform(train_data.X)
    test_data.X = minmax.transform(test_data.X)

    batch_size = 64
    dim_in = train_data.num_features

    num_hiddens = [10, 10]
    model = FNN(dim_in, 2, num_hiddens, epsilon=1e-2, bias_epsilon=1e-1)

    # cfx_data = dataset.Custom_Dataset("../data/german_train_lim.csv", "credit_risk")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    # NOTE: we need shuffle=False for CFX's to be in right order (or need to change how we generate CFX)
    # There should be dataset loader that returns index of the data point
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
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
            cfx_x, is_cfx = cfx_generator.run_wachter(scaler=minmax)
        model.train()
        total_loss = 0
        for batch, (X, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            if cfx_x is None:
                loss = model.get_loss(X, y, None, None, 0)
            else:
                this_cfx = cfx_x[batch * batch_size: (batch + 1) * batch_size]
                this_is_cfx = is_cfx[batch * batch_size: (batch + 1) * batch_size]
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
                for X, y in test_dataloader:
                    y_pred = model.forward_point_weights_bias(X.float()).argmax(dim=-1)
                    acc_cnt += torch.sum(y_pred == y).item()
            print("Epoch", str(epoch), "accuracy:", acc_cnt / len(test_data))
        model.eval()
        with torch.no_grad():
            # make sure CFX is valid
            if cfx_x is not None:
                for batch_id, (X, y) in enumerate(train_dataloader):
                    i = batch_id * batch_size
                    cfx_y = model.forward_point_weights_bias(cfx_x[i:i + batch_size]).argmax(dim=-1)
                    is_cfx[i:i + batch_size] = is_cfx[i:i + batch_size] & (cfx_y == (1 - y)).bool()

    model.eval()
    with torch.no_grad():
        # eval on train for right now since we have those CFX
        total_samples, correct = 0, 0
        for X, y in train_dataloader:
            total_samples += len(X)
            X = X.float()
            y_pred = model.forward_point_weights_bias(X).argmax(dim=-1)
            correct += torch.sum((y_pred == y).float()).item()
            # If we want CFX accuracy, need to re-run CFX generation
            # cfx_y_pred = model.forward_point_weights_bias(cfx_data.X).argmax(dim=-1)
            # print("Train CFX Acc:", torch.mean((cfx_y_pred == (1 - y)).float()).item())
        print("Train accuracy: ", round(correct / total_samples, 4))
        total_samples, correct = 0, 0
        for X, y in test_dataloader:
            total_samples += len(X)
            X = X.float()
            y_pred = model.forward_point_weights_bias(X).argmax(dim=-1)
            correct += torch.sum((y_pred == y).float()).item()
        print("Test accuracy: ", round(correct / total_samples, 4))
