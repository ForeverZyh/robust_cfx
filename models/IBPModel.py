import torch

import torch.nn as nn
import torch.nn.functional as F
from auto_LiRPA import BoundedModule, BoundedParameter
from auto_LiRPA.perturbations import *
from utils.utilities import seed_everything, FAKE_INF, EPS


class VerifyModel:
    def __init__(self, model, dummy_input):
        self.ori_model = model
        self.model = BoundedModule(model, dummy_input, bound_opts={
            'sparse_intermediate_bounds': False,
            'sparse_conv_intermediate_bounds': False,
            'sparse_intermediate_bounds_with_ibp': False})
        self.final_node = self.model.final_name
        self.root_nodes = set(self.model.root_names)
        self.loss_func = nn.BCELoss()

    def forward_point_weights_bias(self, x):
        return self.model.forward(x)

    def forward_IBP(self, x, forward_first=False):
        """
        :param x: the input
        :param forward_first: True or False, if True, then forward the input before IBP, otherwise, directly call IBP
        The IBP is required to be called after the forward pass during training.
        However, in most cases, the self.forward() is already called before IBP, so we can skip the forward pass in IBP
        :return: the lower and upper bound for the output
        """
        c = torch.tensor([[[-1, 1]] for _ in range(x.shape[0])]).float()
        if forward_first:
            _ = self.model.forward(x)
            return self.model(method_opt="compute_bounds", C=c, method="IBP", final_node_name=None, no_replicas=True)
        return self.model(x=(x,), method_opt="compute_bounds", C=c, method="IBP", final_node_name=None,
                          no_replicas=True)

    def forward_CROWN_IBP(self, x, return_A, forward_IBP_first=False, forward_first=False):
        """
        :param x: the input
        :param return_A: whether to return lA and uA
        A[node_1]=set(list(node_2)) compute the A matrix for node_1 with respect to node_2
        :param forward_IBP_first: True or False, if True, then forward with IBP the input before CROWN_IBP;
         otherwise, directly call CROWN_IBP
        :param forward_first: True or False, if True, then forward the input before IBP, otherwise, directly call IBP
        :return:
        """
        if forward_IBP_first:
            _ = self.forward_IBP(x, forward_first=forward_first)
        c = torch.tensor([[[-1, 1]] for _ in range(x.shape[0])]).float()
        return self.model(method_opt="compute_bounds", C=c, method="CROWN-IBP",
                          bound_lower=True, bound_upper=True, final_node_name=None, average_A=True,
                          no_replicas=True, return_A=return_A, needed_A_dict={self.final_node: self.root_nodes})

    def smash_A(self, A_dict):
        """
        Smash the A dict into a vector
        """
        ret = {"lA": [], "lW": [], "uA": [], "uW": [], "lbias": 0, "ubias": 0}
        A_dict = A_dict[self.final_node]
        for node in self.model.nodes():
            if node.name in A_dict:
                batch_size = A_dict[node.name]['lA'].shape[0]
                ret["lA"].append(A_dict[node.name]['lA'].view(batch_size, -1))
                ret["lW"].append(node.lower.view(1, -1))
                ret["uA"].append(A_dict[node.name]['uA'].view(batch_size, -1))
                ret["uW"].append(node.upper.view(1, -1))
                # they are the same across all nodes.
                ret["lbias"] = A_dict[node.name]['lbias']
                ret["ubias"] = A_dict[node.name]['ubias']

        for key in ["lA", "lW", "uA", "uW"]:
            ret[key] = torch.cat(ret[key], dim=1)

        return ret

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    @staticmethod
    def get_ub(k, k_1, b, b_1, w_lb, w_ub):
        '''
        return max(k_1 w + b_1) subject to k w + b >= 0, for w in [w_lb, w_ub]
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

    def get_diffs_binary(self, x, cfx_x, y, forward_first=False):
        '''
        x : data
        cfx_x : the counterfactuals for x
        y : ground-truth labels for x

        Return a tuple:
            - First entry: boolean tensor where each entry is
                - True the original prediction can be valid, i.e., the original prediction can be correct for any one
                of the parameter shifts
                - False otherwise
                Only used for unittests now
            - Second entry: real-valued tensor where each entry is
                - Lower bound of CFX output when x's ground truth is 0
                - Upper bound of CFX output when x's ground truth is 1

        '''
        ilb, iub = self.forward_IBP(x, forward_first=forward_first)
        lb, ub, ret_A = self.forward_CROWN_IBP(x, forward_first=forward_first, return_A=True)
        iclb, icub = self.forward_IBP(cfx_x, forward_first=forward_first)
        clb, cub, ret_cA = self.forward_CROWN_IBP(cfx_x, forward_first=forward_first, return_A=True)
        lb = torch.max(ilb, lb)  # CROWN-IBP is not guaranteed to be tighter than IBP
        ub = torch.min(iub, ub)
        clb = torch.max(iclb, clb)
        cub = torch.min(icub, cub)

        lb = lb.squeeze(-1)
        ub = ub.squeeze(-1)
        clb = clb.squeeze(-1)
        cub = cub.squeeze(-1)

        ret_A = self.smash_A(ret_A)
        ret_cA = self.smash_A(ret_cA)
        cfx_output_lb = -self.get_ub(-ret_A["lA"], -ret_cA["lA"], -ret_A["lbias"].view(-1), -ret_cA["lbias"].view(-1),
                                     ret_A["lW"], ret_A["uW"])
        cfx_output_ub = self.get_ub(ret_A["uA"], ret_cA["uA"], ret_A["ubias"].view(-1), ret_cA["ubias"].view(-1),
                                    ret_A["lW"], ret_A["uW"])
        cfx_output_lb = torch.max(cfx_output_lb, clb)
        cfx_output_ub = torch.min(cfx_output_ub, cub)
        can_cfx_pred_invalid = torch.where(y == 0, (lb <= 0) & (cfx_output_lb <= 0), (ub > 0) & (cfx_output_ub > 0))

        return can_cfx_pred_invalid, torch.where(y == 0, cfx_output_lb, cfx_output_ub)

    def get_diffs_binary_crownibp(self, x, cfx_x, y, forward_first=False):
        ilb, iub = self.forward_IBP(x, forward_first=forward_first)
        lb, ub = self.forward_CROWN_IBP(x, return_A=False)
        iclb, icub = self.forward_IBP(cfx_x, forward_first=forward_first)
        clb, cub = self.forward_CROWN_IBP(cfx_x, return_A=False)
        lb = torch.max(ilb, lb)  # CROWN-IBP is not guaranteed to be tighter than IBP
        ub = torch.min(iub, ub)
        clb = torch.max(iclb, clb)
        cub = torch.min(icub, cub)

        lb = lb.squeeze(-1)
        ub = ub.squeeze(-1)
        clb = clb.squeeze(-1)
        cub = cub.squeeze(-1)
        can_cfx_pred_invalid = torch.where(y == 0, (lb <= 0) & (clb <= 0), (ub > 0) & (cub > 0))
        return can_cfx_pred_invalid, torch.where(y == 0, clb, cub)

    def get_diffs_binary_ibp(self, x, cfx_x, y, forward_first=False):
        lb, ub = self.forward_IBP(x, forward_first=forward_first)
        clb, cub = self.forward_IBP(cfx_x, forward_first=forward_first)
        lb = lb.squeeze(-1)
        ub = ub.squeeze(-1)
        clb = clb.squeeze(-1)
        cub = cub.squeeze(-1)
        can_cfx_pred_invalid = torch.where(y == 0, (lb <= 0) & (clb <= 0), (ub > 0) & (cub > 0))
        return can_cfx_pred_invalid, torch.where(y == 0, clb, cub)

    def get_loss(self, x, y, cfx_x, is_cfx, lambda_ratio=1.0, loss_type="ours"):
        '''
            x is data
            y is ground-truth label
        '''
        # convert x to float32
        assert loss_type in ["ours", "ibp", "crownibp"]
        x = x.float()
        ori_output = self.forward_point_weights_bias(x)
        # print(ori_output)
        ori_output = torch.sigmoid(ori_output[:, 1] - ori_output[:, 0])
        ori_loss = self.loss_func(ori_output, y.float())
        if lambda_ratio == 0:
            return ori_loss.mean()
        # print(ori_loss)

        if loss_type == "ours":
            _, cfx_output = self.get_diffs_binary(x, cfx_x, y)
        elif loss_type == "ibp":
            _, cfx_output = self.get_diffs_binary_ibp(x, cfx_x, y)
        elif loss_type == "crownibp":
            _, cfx_output = self.get_diffs_binary_crownibp(x, cfx_x, y)
        # if the cfx is valid and the original prediction is correct, then we use the cfx loss
        is_real_cfx = is_cfx & torch.where(y == 0, ori_output <= 0.5, ori_output > 0.5)
        # print(is_real_cfx, cfx_output)
        cfx_output = torch.sigmoid(cfx_output)
        cfx_loss = self.loss_func(cfx_output, 1.0 - y)
        # print(cfx_loss)
        cfx_loss = torch.where(is_real_cfx, cfx_loss, 0)
        # print(cfx_loss)
        return (ori_loss + lambda_ratio * cfx_loss).mean()


class BoundedLinear(nn.Module):
    def __init__(self, input_dim, out_dim, epsilon, bias_epsilon, norm=np.inf):
        super(BoundedLinear, self).__init__()
        self.epsilon = epsilon
        self.bias_epsilon = bias_epsilon
        self.ptb_weight = PerturbationLpNorm(norm=norm, eps=epsilon)
        self.ptb_bias = PerturbationLpNorm(norm=norm, eps=bias_epsilon)
        self.linear = nn.Linear(input_dim, out_dim)
        self.linear.weight = BoundedParameter(self.linear.weight.data, self.ptb_weight)
        self.linear.bias = BoundedParameter(self.linear.bias.data, self.ptb_bias)

    def forward(self, x):
        return self.linear(x)

    def forward_with_noise(self, x, cfx):
        # only for test use
        layer = self.linear
        weights = layer.weight.data + (
                0.5 - torch.rand(layer.weight.data.shape)) * 2 * self.epsilon
        bias = layer.bias.data + (
                0.5 - torch.rand(layer.bias.data.shape)) * 2 * self.bias_epsilon
        x = F.linear(x, weights, bias)
        cfx = F.linear(cfx, weights, bias)
        return x, cfx


class LinearBlock(nn.Module):
    def __init__(self, input_dim, out_dim, epsilon, bias_epsilon, activation, dropout):
        super().__init__()
        self.linear = BoundedLinear(input_dim, out_dim, epsilon, bias_epsilon)
        if dropout == 0:
            self.block = activation()
        else:
            self.block = nn.Sequential(
                activation(),
                nn.Dropout(dropout),
            )

    def forward(self, x):
        x = self.linear(x)
        return self.block(x)

    def forward_with_noise(self, x, cfx):
        x, cfx = self.linear.forward_with_noise(x, cfx)
        return self.block(x), self.block(cfx)


class MultilayerPerception(nn.Module):
    def __init__(self, dims, epsilon, bias_epsilon, activation, dropout):
        super().__init__()
        layers = []
        num_blocks = len(dims)
        self.blocks = []
        for i in range(1, num_blocks):
            self.blocks.append(LinearBlock(dims[i - 1], dims[i], epsilon, bias_epsilon, activation, dropout=dropout))
        self.model = nn.Sequential(*self.blocks)

    def forward(self, x):
        return self.model(x)

    def forward_with_noise(self, x, cfx_x):
        x = x, cfx_x
        for block in self.blocks:
            x = block.forward_with_noise(*x)
        return x


class FNN(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens: list, epsilon=0.0, bias_epsilon=0.0, activation=nn.ReLU,
                 dropout=0):
        super(FNN, self).__init__()
        self.num_hiddens = num_hiddens
        self.activation = activation
        self.dropout = dropout
        dims = [num_inputs] + num_hiddens
        self.encoder = MultilayerPerception(dims, epsilon, bias_epsilon, activation, dropout=dropout)
        self.final_fc = BoundedLinear(num_hiddens[-1], num_outputs, epsilon, bias_epsilon)
        self.epsilon = epsilon
        self.bias_epsilon = bias_epsilon

    def forward(self, x):
        x = x.float()  # necessary for Counterfactual generation to work
        x = self.encoder(x)
        x = self.final_fc(x)
        return x

    def forward_with_noise(self, x, cfx_x):
        x = x.float(), cfx_x.float()
        x = self.encoder.forward_with_noise(*x)
        x = self.final_fc.forward_with_noise(*x)
        return x


# class CounterNet(IBPModel):
#     def __init__(self, num_inputs, num_outputs, num_hiddens: list, epsilon=0.0, bias_epsilon=0.0, activation=F.relu):
#         super(CounterNet, self).__init__(num_inputs, num_outputs)
#         self.encoder_net = FNN(num_inputs, num_outputs, num_hiddens, epsilon, bias_epsilon, activation)
#
#     def forward_except_last(self, x):
#         return self.encoder_net.forward_except_last(x)
#
#     def forward_point_weights_bias(self, x):
#         return self.forward_point_weights_bias(x)
#

if __name__ == '__main__':
    seed_everything(0)
    batch_size = 100
    dim_in = 2
    model_ori = FNN(dim_in, 2, [2, 4], epsilon=1e-1, bias_epsilon=1e-2, activation=lambda: nn.LeakyReLU(0.1))
    x = torch.normal(0, 0.2, (batch_size, dim_in))
    x[:batch_size // 2, 0] = x[:batch_size // 2, 0] + 2
    cfx_x = x + torch.normal(0, 0.2, x.shape)
    cfx_x[:batch_size // 2, :] = cfx_x[:batch_size // 2, :] - 1
    cfx_x[batch_size // 2:, :] = cfx_x[batch_size // 2:, :] + 1
    y = torch.ones((batch_size,))
    y[:batch_size // 2] = 0
    print(f"Centered at (2, 0) with label {y[0].long().item()}", x[:5])
    print(f"CFX of the above points Centered at (1, -1) with label {(1 - y[0]).long().item()}", cfx_x[:5])
    print(f"Centered at (0, 0) with label {y[-1].long().item()}", x[-5:])
    print(f"CFX of the above points Centered at (1, 1) with label {(1 - y[-1]).long().item()}", cfx_x[-5:])
    model = VerifyModel(model_ori, x[2:])
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for epoch in range(500):
        optimizer.zero_grad()
        loss = model.get_loss(x, y, cfx_x, torch.ones(cfx_x.shape[0]).bool(),
                              0.1, loss_type="ours")  # change lambda_ratio to 0.0 results in low CFX accuracy.
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(loss)

    model.eval()
    with torch.no_grad():
        y_pred = model.forward_point_weights_bias(x).argmax(dim=-1)
        print("Acc:", torch.mean((y_pred == y).float()).item())
        cfx_y_pred = model.forward_point_weights_bias(cfx_x).argmax(dim=-1)
        print("CFX Acc:", torch.mean((cfx_y_pred == (1 - y)).float()).item())
