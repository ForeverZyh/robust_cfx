import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from ibp import LinearBound, activation, IntervalBoundedTensor
import torch.nn.functional as F

sys.path.append("..")
from utils import dataset

FAKE_INF = 10
EPS = 1e-8


class IBPModel(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(IBPModel, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs


class FNN(IBPModel):
    def __init__(self, num_inputs, num_outputs, num_hiddens: list, epsilon=0.0, bias_epsilon=0.0, activation=F.relu):
        super(FNN, self).__init__(num_inputs, num_outputs)
        self.num_hiddens = num_hiddens
        self.activation = activation
        for i, num_hidden in enumerate(num_hiddens):
            setattr(self, 'fc{}'.format(i),
                    LinearBound(num_inputs, num_hidden, epsilon=epsilon, bias_epsilon=bias_epsilon))
            num_inputs = num_hidden
        self.fc_final = LinearBound(num_inputs, num_outputs, epsilon=epsilon, bias_epsilon=bias_epsilon)
        self.epsilon = epsilon
        self.bias_epsilon = bias_epsilon
        self.loss_func = torch.nn.BCELoss(reduce=False)

    def forward_except_last(self, x):
        for i, num_hidden in enumerate(self.num_hiddens):
            x = getattr(self, 'fc{}'.format(i))(x)
            x = activation(self.activation, x)
        return x

    def forward_point_weights_bias(self, x):
        for i, num_hidden in enumerate(self.num_hiddens):
            x = getattr(self, 'fc{}'.format(i)).forward_point_weights_bias(x)
            x = activation(self.activation, x)
        return self.fc_final.forward_point_weights_bias(x)

    def forward(self, x):
        x = self.forward_except_last(x)
        x = self.fc_final(x)
        return x

    def get_lb_ub_bound(self, x: IntervalBoundedTensor, W, b):
        W_lb = W - self.epsilon * 2
        W_ub = W + self.epsilon * 2
        left_end_points = x * W_lb
        right_end_points = x * W_ub

        def get_k_and_b(left, right, b_):
            k = (right - left) / (self.epsilon * 4)
            b = left - k * W_lb
            return k, b.sum(dim=-1) + b_

        return get_k_and_b(left_end_points.lb, right_end_points.lb, b - self.bias_epsilon), \
               get_k_and_b(left_end_points.ub, right_end_points.ub, b + self.bias_epsilon)

    @staticmethod
    def get_ub(k, k_1, b, b_1, w_lb, w_ub):
        w_ret = torch.where((k < 0) | ((k == 0) & (k_1 < 0)), w_lb, w_ub)
        ret = torch.sum(w_ret * k_1, dim=-1) + b_1
        t = torch.sum(w_ret * k, dim=-1) + b
        sorted_value = torch.where(k * k_1 >= 0, 0, torch.abs(k_1) / (EPS + torch.abs(k)))
        t_delta = torch.abs(k) * (w_ub - w_lb)
        t_1_delta = torch.abs(k_1) * (w_ub - w_lb)
        sorted_value, sorted_indices = sorted_value.sort(dim=-1, descending=True)
        # reorder t_delta by stored indices
        t_delta = t_delta.gather(dim=-1, index=sorted_indices)
        t_1_delta = t_1_delta.gather(dim=-1, index=sorted_indices)
        t_delta_sum = t_delta.cumsum(dim=-1)
        delta = t.unsqueeze(-1) - t_delta_sum
        percent = torch.where(delta < 0, torch.clamp(delta / (t_delta + EPS), -1, 0) + 1, 1)
        percent = torch.where(sorted_value > 0, percent, 0)
        # t_1_delta = t_1_delta.cumsum(dim=-1) * (sorted_value > 0).float()
        ret = ret + torch.sum(t_1_delta * percent, dim=-1)
        return torch.where(t >= 0, ret, -FAKE_INF)

    def get_diffs_binary(self, x, cfx_x, y):
        x = self.forward_except_last(x)
        cfx_x = self.forward_except_last(cfx_x)
        # print(x)
        # print(cfx_x)
        # print(y)
        # print(self.fc_final.forward_point_weights_bias(x.val))
        # print(self.fc_final.forward_point_weights_bias(cfx_x.val))
        # cfx_y = 1 - y
        y_final_weights = self.fc_final.linear.weight[0, :] - self.fc_final.linear.weight[1, :]
        y_final_bias = self.fc_final.linear.bias[0] - self.fc_final.linear.bias[1]
        (alpha_k, alpha_b), (beta_k, beta_b) = self.get_lb_ub_bound(x, y_final_weights, y_final_bias)
        y_final_weights_ub = y_final_weights + 2 * self.epsilon
        y_final_weights_lb = y_final_weights - 2 * self.epsilon
        (alpha_k_1, alpha_b_1), (beta_k_1, beta_b_1) = self.get_lb_ub_bound(cfx_x, y_final_weights, y_final_bias)
        cfx_output_lb = -self.get_ub(-alpha_k, -alpha_k_1, -alpha_b, -alpha_b_1, y_final_weights_lb,
                                     y_final_weights_ub)
        cfx_output_ub = self.get_ub(beta_k, beta_k_1, beta_b, beta_b_1, y_final_weights_lb,
                                    y_final_weights_ub)
        # print(torch.any(beta_k * beta_k_1 < 0))
        # print(torch.any(alpha_k * alpha_k_1 < 0))
        is_real_cfx = torch.where(y == 0, cfx_output_lb <= 0, cfx_output_ub >= 0)
        return is_real_cfx, torch.where(y == 0, cfx_output_lb, cfx_output_ub)

    def get_loss(self, x, cfx_x, y, lambda_ratio=1.0):
        ori_output = model.forward_point_weights_bias(x)
        # print(ori_output)
        ori_output = torch.sigmoid(ori_output[:, 0] - ori_output[:, 1])
        ori_loss = self.loss_func(ori_output, y.float())
        if lambda_ratio == 0:
            return ori_loss.mean()
        # print(ori_loss)
        is_real_cfx, cfx_output = self.get_diffs_binary(x, cfx_x, y)
        # print(is_real_cfx, cfx_output)
        cfx_output = torch.sigmoid(cfx_output)
        cfx_loss = self.loss_func(cfx_output, 1.0 - y)
        # print(cfx_loss)
        cfx_loss = torch.where(is_real_cfx, cfx_loss, 0)
        # print(cfx_loss)
        return (ori_loss + lambda_ratio * cfx_loss).mean()


if __name__ == '__main__':
    torch.random.manual_seed(0)

    train_data = dataset.Custom_Dataset("../data/german_train.csv", "credit_risk")
    test_data = dataset.Custom_Dataset("../data/german_test.csv", "credit_risk")
    # cast train_data.X to torch
    train_data.X = torch.from_numpy(train_data.X).float()

    batch_size = len(train_data) # to make it easy for now
    dim_in = train_data.num_features

    model = FNN(dim_in, 2, [2, 4], epsilon=1e-2, bias_epsilon=1e-1)

    cfx_data = dataset.Custom_Dataset("../data/german_train.csv", "credit_risk")
    cfx_data.X = cfx_data.X + np.random.normal(0, 0.2, cfx_data.X.shape)
    cfx_data.X[:batch_size // 2, :] = cfx_data.X[:batch_size // 2, :] - 1
    cfx_data.X[batch_size // 2:, :] = cfx_data.X[batch_size // 2:, :] + 1
    # cast CFX data X to torch
    cfx_data.X = torch.from_numpy(cfx_data.X).float()

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # x = torch.normal(0, 0.2, (batch_size, dim_in))
    # x[:batch_size // 2, 0] = x[:batch_size // 2, 0] + 2
    # cfx_x = x + torch.normal(0, 0.2, x.shape)
    # cfx_x[:batch_size // 2, :] = cfx_x[:batch_size // 2, :] - 1
    # cfx_x[batch_size // 2:, :] = cfx_x[batch_size // 2:, :] + 1

    # y = torch.ones((batch_size,))
    # y[:batch_size // 2] = 0
    # print(f"Centered at (2, 0) with label {y[0].long().item()}", x[:5])
    # print(f"CFX of the above points Centered at (1, -1) with label {(1- y[0]).long().item()}", cfx_x[:5])
    # print(f"Centered at (0, 0) with label {y[-1].long().item()}", x[-5:])
    # print(f"CFX of the above points Centered at (1, 1) with label {(1- y[-1]).long().item()}", cfx_x[-5:])
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for epoch in range(500):
        for batch, (X, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss = model.get_loss(X, cfx_data.X, y, 0.1)  # change lambda_ratio to 0.0 results in low CFX accuracy.
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(loss)

    model.eval()
    with torch.no_grad():
        # eval on train for right now since we have those CFX
        for X,y in train_dataloader:
            y_pred = model.forward_point_weights_bias(X).argmin(dim=-1)
            print("Acc:", torch.mean((y_pred == y).float()).item())
            cfx_y_pred = model.forward_point_weights_bias(cfx_data.X).argmin(dim=-1)
            print("CFX Acc:", torch.mean((cfx_y_pred == (1 - y)).float()).item())
