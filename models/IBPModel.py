import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from ibp import LinearBound, activation, IntervalBoundedTensor
import torch.nn.functional as F

sys.path.append("..")
from utils import dataset
from utils import cfx

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
        x = x.float() # necessary for Counterfactual generation to work
        x = self.forward_except_last(x)
        x = self.fc_final(x)
        return x

    def get_lb_ub_bound(self, x: IntervalBoundedTensor, W, b):
        '''
        x : embedding of data (all layers except last FC layer). Dimension m x 1
        W : W_0 - W_1 where W was the 2 x m final FC layer
        b : b_0 - b_1 where b was the 2 x 1 final bias vector

        returns alpha, beta, alpha', beta' s.t. 
             - logits y for x satisfy y_0 - y_1 <= w * alpha + beta + b
             - logits y' for x' satisfy y'_0 - y'_1 <= w * alpha' + beta' + b
        '''
        # Add/subtract 2eps because we have to account for eps for each W_0 and W_1
        W_lb = W - self.epsilon * 2
        W_ub = W + self.epsilon * 2
        
        left_end_points = x * W_lb # range of Wx when W is minimized
        right_end_points = x * W_ub

        def get_k_and_b(left, right, b_):
            k = (right - left) / (self.epsilon * 4)
            b = left - k * W_lb
            return k, b.sum(dim=-1) + b_

        return get_k_and_b(left_end_points.lb, right_end_points.lb, b - self.bias_epsilon), \
               get_k_and_b(left_end_points.ub, right_end_points.ub, b + self.bias_epsilon)

    @staticmethod
    def get_ub(k, k_1, b, b_1, w_lb, w_ub):
        '''
        kx+b is an upper bound for Wx+b  
        k_1 x' + b_1 is an upper bound for Wx'+b
        w_lb, w_ub are lower and upper bounds for W (final FC layer)
        '''
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

    def get_diffs_binary(self, x, cfx_x, is_cfx, y):
        '''
        x : data
        cfx_x : the counterfactuals for x
        is_cfx : 0 if we failed to find a CFX for x_i (i.e., cfx_x[i]=0 but should be None)
                 1 otherwise 
        y : ground-truth labels for x

        Return a tuple:
            - First entry: boolean tensor where each entry is 
                - True the counterfactual is INVALID, i.e., if either
                    - x's ground truth is 0 and CFX output's lower bound is less than 0
                    - x's ground truth is 1 and CFX output's upper bound is greater than 0
                - False otherwise
            - Second entry: real-valued tensor where each entry is 
                - Lower bound of CFX output when x's ground truth is 0
                - Upper bound of CFX output when x's ground truth is 1 

        '''
        embed_x = self.forward_except_last(x)
        embed_cfx_x = self.forward_except_last(cfx_x)
        # so we can write logits y_0-y_1 = y_final_weights * embed_x + y_final_bias
        y_final_weights = self.fc_final.linear.weight[0, :] - self.fc_final.linear.weight[1, :]
        y_final_bias = self.fc_final.linear.bias[0] - self.fc_final.linear.bias[1]
        (alpha_k, alpha_b), (beta_k, beta_b) = self.get_lb_ub_bound(embed_x, y_final_weights, y_final_bias)
        y_final_weights_ub = y_final_weights + 2 * self.epsilon
        y_final_weights_lb = y_final_weights - 2 * self.epsilon
        (alpha_k_1, alpha_b_1), (beta_k_1, beta_b_1) = self.get_lb_ub_bound(embed_cfx_x, y_final_weights, y_final_bias)
        cfx_output_lb = -self.get_ub(-alpha_k, -alpha_k_1, -alpha_b, -alpha_b_1, y_final_weights_lb,
                                     y_final_weights_ub)
        cfx_output_ub = self.get_ub(beta_k, beta_k_1, beta_b, beta_b_1, y_final_weights_lb,
                                    y_final_weights_ub)
        # print(torch.any(beta_k * beta_k_1 < 0))
        # print(torch.any(alpha_k * alpha_k_1 < 0))
        is_real_cfx = torch.where(y == 0, cfx_output_lb <= 0, cfx_output_ub >= 0)
        # if CFX was fake (i.e., None represented by [0]), then fix is_real_cfx to reflect this
        is_real_cfx = torch.where(is_cfx, is_real_cfx, False)

        return is_real_cfx, torch.where(y == 0, cfx_output_lb, cfx_output_ub)

    def get_loss(self, x, y, cfx_x, is_cfx, lambda_ratio=1.0):
        '''
            x is data
            y is ground-truth label
        '''
        # convert x to float32
        x = x.float()
        ori_output = model.forward_point_weights_bias(x)
        # print(ori_output)
        ori_output = torch.sigmoid(ori_output[:, 0] - ori_output[:, 1])
        ori_loss = self.loss_func(ori_output, y.float())
        if lambda_ratio == 0:
            return ori_loss.mean()
        # print(ori_loss)
            
        is_real_cfx, cfx_output = self.get_diffs_binary(x, cfx_x, is_cfx, y)
        # print(is_real_cfx, cfx_output)
        cfx_output = torch.sigmoid(cfx_output)
        cfx_loss = self.loss_func(cfx_output, 1.0 - y)
        # print(cfx_loss)
        cfx_loss = torch.where(is_real_cfx, cfx_loss, 0)
        # print(cfx_loss)
        return (ori_loss + lambda_ratio * cfx_loss).mean()



if __name__ == '__main__':
    torch.random.manual_seed(0)
    train_data = dataset.Custom_Dataset("../data/german_train_lim.csv", "credit_risk")
    test_data = dataset.Custom_Dataset("../data/german_test.csv", "credit_risk")
    minmax = MinMaxScaler(clip=True)
    train_data.X = minmax.fit_transform(train_data.X)
    test_data.X = minmax.transform(test_data.X)

    batch_size = 10 
    dim_in = train_data.num_features

    num_hiddens = [10, 10]
    model = FNN(dim_in, 2, num_hiddens, epsilon=1e-2, bias_epsilon=1e-1)
    
    # cfx_data = dataset.Custom_Dataset("../data/german_train_lim.csv", "credit_risk")
    # cfx_data.X = cfx_data.X + torch.normal(0, 0.2, cfx_data.X.shape)
    # cfx_data.X[:batch_size // 2, :] = cfx_data.X[:batch_size // 2, :] - 1
    # cfx_data.X[batch_size // 2:, :] = cfx_data.X[batch_size // 2:, :] + 1
    # # cast CFX data X to torch
    # cfx_data.X = torch.from_numpy(cfx_data.X).float()


    
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    # NOTE: we need shuffle=False for CFX's to be in right order (or need to change how we generate CFX)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    @torch.no_grad()
    def predictor(X: np.ndarray) -> np.ndarray:
        X = torch.as_tensor(X, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            ret = model.forward_point_weights_bias(X)
            ret = F.softmax(ret, dim=1)
            ret = ret.cpu().numpy()
            # print(ret)
        model.train()
        return ret
    

    
    cfx_generator = cfx.CFX_Generator(predictor, train_data, num_layers = len(num_hiddens))
    # warm up
    model.train()
    for epoch in range(50):
        for batch, (X, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss = model.get_loss(X, y, None, None, 0)
            loss.backward()
            optimizer.step()
        model.eval()
        acc_cnt = 0
        with torch.no_grad():
            for X, y in test_dataloader:
                y_pred = model.forward_point_weights_bias(X.float()).argmin(dim=-1)
                acc_cnt += torch.sum(y_pred == y).item()
        if epoch % 10 == 0:
            print("Epoch",str(epoch),"accuracy:", acc_cnt / len(test_data))

    model.train()
    for epoch in range(5):
        if epoch % 10 == 0:
            # generate CFX
            cfx_x, is_cfx = cfx_generator.run_wachter(scaler=minmax)
        for batch, (X, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            this_cfx = cfx_x[batch * batch_size : (batch + 1) * batch_size]
            this_is_cfx = is_cfx[batch * batch_size : (batch + 1) * batch_size]
            loss = model.get_loss(X, y, this_cfx, this_is_cfx, 0.1)  # changing lambda_ratio to 0.0 results in low CFX accuracy.
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            print("Epoch",str(epoch),"loss:",loss)

    model.eval()
    with torch.no_grad():
        # eval on train for right now since we have those CFX
        total_samples,correct = 0,0
        for X,y in train_dataloader:
            total_samples += len(X)
            X = X.float()
            y_pred = model.forward_point_weights_bias(X).argmin(dim=-1)
            correct += torch.sum((y_pred == y).float()).item()
            # If we want CFX accuracy, need to re-run CFX generation
            # cfx_y_pred = model.forward_point_weights_bias(cfx_data.X).argmin(dim=-1)
            # print("Train CFX Acc:", torch.mean((cfx_y_pred == (1 - y)).float()).item())
        print("Train accuracy: ", round(correct / total_samples, 4))
        total_samples, correct = 0, 0
        for X,y in test_dataloader:
            total_samples += len(X)
            X = X.float()
            y_pred = model.forward_point_weights_bias(X).argmin(dim=-1)
            correct += torch.sum((y_pred == y).float()).item()
        print("Test accuracy: ", round(correct / total_samples, 4))
