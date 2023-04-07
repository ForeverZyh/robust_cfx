import torch

from models.ibp import LinearBound, activation, IntervalBoundedTensor
import torch.nn.functional as F


class IBPModel(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(IBPModel, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs


class FNN(IBPModel):
    def __init__(self, num_inputs, num_outputs, num_hiddens: list, epsilon=0.0, bias_epsilon=0.0):
        super(FNN, self).__init__(num_inputs, num_outputs)
        self.num_hiddens = num_hiddens
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
            x = activation(F.relu, x)
        return x

    def forward_point_weights_bias(self, x):
        for i, num_hidden in enumerate(self.num_hiddens):
            x = getattr(self, 'fc{}'.format(i)).forward_point_weights_bias(x)
            x = activation(F.relu, x)
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
        output_ub = (beta_k * y_final_weights_ub).sum(dim=-1) + beta_b
        output_lb = (alpha_k * y_final_weights_lb).sum(dim=-1) + alpha_b
        (alpha_k_1, alpha_b_1), (beta_k_1, beta_b_1) = self.get_lb_ub_bound(cfx_x, y_final_weights, y_final_bias)
        cfx_output_ub = (beta_k_1 * y_final_weights_ub).sum(dim=-1) + beta_b_1
        cfx_output_lb = (alpha_k_1 * y_final_weights_lb).sum(dim=-1) + alpha_b_1
        is_real_cfx = torch.where(y == 0, (output_lb <= 0) & (cfx_output_lb <= 0),
                                  (output_ub >= 0) & (cfx_output_ub >= 0))
        return is_real_cfx, torch.where(y == 0, cfx_output_lb, cfx_output_ub)

    def get_loss(self, x, cfx_x, y, lambda_ratio=1.0):
        ori_output = model.forward_point_weights_bias(x)
        ori_output = torch.sigmoid(ori_output[:, 0] - ori_output[:, 1])
        ori_loss = self.loss_func(ori_output, y.float())
        # print(ori_loss)
        is_real_cfx, cfx_output = self.get_diffs_binary(x, cfx_x, y)
        cfx_output = torch.sigmoid(cfx_output)
        cfx_loss = self.loss_func(cfx_output, 1.0 - y)
        # print(cfx_loss)
        cfx_loss = cfx_loss * is_real_cfx.float()
        # print(cfx_loss)
        return (ori_loss + lambda_ratio * cfx_loss).mean()


if __name__ == '__main__':
    torch.random.manual_seed(0)
    batch_size = 100
    dim_in = 2
    model = FNN(dim_in, 2, [2, 4], epsilon=1e-2, bias_epsilon=1e-1)
    x = torch.normal(0, 0.2, (batch_size, dim_in))
    x[:batch_size // 2, 0] = x[:batch_size // 2, 0] + 2
    cfx_x = x + torch.normal(0, 0.2, x.shape)
    cfx_x[:batch_size // 2, 0] = cfx_x[:batch_size // 2, 0] - 1.5
    cfx_x[batch_size // 2:, 0] = cfx_x[batch_size // 2:, 0] + 1.5
    y = torch.ones((batch_size,))
    y[:batch_size // 2] = 0
    print(x[:10], cfx_x[:10], y[:10])
    print(x[-10:], cfx_x[-10:], y[-10:])
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for epoch in range(100):
        optimizer.zero_grad()
        loss = model.get_loss(x, cfx_x, y, 0.5)  # change lambda_ratio to 0.0 results in low CFX accuracy.
        loss.backward()
        optimizer.step()
        print(loss)

    model.eval()
    with torch.no_grad():
        y_pred = model.forward_point_weights_bias(x).argmin(dim=-1)
        print("Acc:", torch.mean((y_pred == y).float()).item())
        cfx_y_pred = model.forward_point_weights_bias(cfx_x).argmin(dim=-1)
        print("CFX Acc:", torch.mean((cfx_y_pred == (1 - y)).float()).item())
