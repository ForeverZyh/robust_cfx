import torch
import torch.nn as nn
import torch.nn.functional as F
from auto_LiRPA import BoundedModule, BoundedParameter
from auto_LiRPA.perturbations import *
from utils.utilities import seed_everything, FAKE_INF, EPS, FNNDims, get_loss_by_type, get_max_loss_by_type
from models.inn import Node, Interval


class VerifyModel(nn.Module):
    def __init__(self, model, dummy_input_shape, loss_func="mse"):
        super(VerifyModel, self).__init__()
        self.ori_model = model
        self.dummy_input_shape = dummy_input_shape
        self.model = BoundedModule(model, torch.zeros(self.dummy_input_shape), bound_opts={
            'sparse_intermediate_bounds': False,
            'sparse_conv_intermediate_bounds': False,
            'sparse_intermediate_bounds_with_ibp': False})
        self.final_node = self.model.final_name
        self.root_nodes = set(self.model.root_names)
        self.loss_func = get_loss_by_type(loss_func)
        self.loss_func_str = loss_func

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

    def get_loss_ori(self, x, y):
        x = x.float()
        ori_output = self.forward_point_weights_bias(x)
        ori_output = torch.sigmoid(ori_output[:, 1] - ori_output[:, 0])
        ori_loss = self.loss_func(ori_output, y.float())
        return ori_loss.mean()

    def get_loss_cfx(self, x, y, cfx_x, is_cfx, lambda_ratio=1.0, loss_type="ours", correct_pred_only=False,
                     valid_cfx_only=False):
        '''
            x is data
            y is ground-truth label
        '''
        # convert x to float32
        assert loss_type in ["ours", "ibp", "crownibp", "none"]
        x = x.float()
        pred = self.forward_point_weights_bias(x)
        y_hat = torch.sigmoid(pred[:, 1] - pred[:, 0]).detach()
        y_hat_hard = pred.argmax(dim=1).detach()
        # print(ori_output)
        if lambda_ratio == 0 or loss_type == "none":
            return 0
        max_loss = get_max_loss_by_type(self.loss_func_str)
        if cfx_x is None:
            return lambda_ratio * max_loss
        # print(ori_loss)

        if loss_type == "ours":
            _, cfx_output = self.get_diffs_binary(x, cfx_x, y_hat_hard)
        elif loss_type == "ibp":
            _, cfx_output = self.get_diffs_binary_ibp(x, cfx_x, y_hat_hard)
        elif loss_type == "crownibp":
            _, cfx_output = self.get_diffs_binary_crownibp(x, cfx_x, y_hat_hard)
        # if the cfx is valid and the original prediction is correct, then we use the cfx loss
        is_real_cfx = torch.ones_like(is_cfx)
        if correct_pred_only:
            is_real_cfx = is_real_cfx & torch.where(y == 0, y_hat <= 0.5, y_hat > 0.5)
        if valid_cfx_only:
            is_real_cfx = is_real_cfx & is_cfx
        # print(is_real_cfx, cfx_output)
        cfx_output = torch.sigmoid(cfx_output)
        cfx_loss = self.loss_func(cfx_output, 1.0 - y_hat)
        # print(cfx_loss)
        cfx_loss = torch.where(is_real_cfx, cfx_loss, max_loss)
        # print(cfx_loss)
        return (lambda_ratio * cfx_loss).mean()

    def save(self, filename):
        filename += ".pt"
        torch.save(self.ori_model.state_dict(), filename)

    def load(self, filename):
        filename += ".pt"
        self.ori_model.load_state_dict(torch.load(filename))
        self.model = BoundedModule(self.ori_model, torch.zeros(self.dummy_input_shape), bound_opts={
            'sparse_intermediate_bounds': False,
            'sparse_conv_intermediate_bounds': False,
            'sparse_intermediate_bounds_with_ibp': False})


class BoundedLinear(nn.Module):
    def __init__(self, input_dim, out_dim, epsilon_ratio, norm=np.inf):
        super(BoundedLinear, self).__init__()
        self.epsilon_ratio = epsilon_ratio
        self.linear = nn.Linear(input_dim, out_dim)
        self.ptb_weight = PerturbationLpNorm(norm=norm, eps=0)
        self.ptb_bias = PerturbationLpNorm(norm=norm, eps=0)
        self.update_eps()
        if self.epsilon_ratio > 0:
            self.linear.weight = BoundedParameter(self.linear.weight.data, self.ptb_weight)
            self.linear.bias = BoundedParameter(self.linear.bias.data, self.ptb_bias)

    def update_eps(self, eps_ratio=None):
        if eps_ratio is None:
            eps_ratio = self.epsilon_ratio
        self.ptb_weight.eps = eps_ratio * self.linear.weight.data.norm(1).item() / self.linear.weight.data.numel()
        self.ptb_bias.eps = eps_ratio * self.linear.bias.data.norm(1).item() / self.linear.bias.data.numel()

    def set_eps_ratio(self, eps_ratio):
        if self.epsilon_ratio > 0:
            self.update_eps(eps_ratio * self.epsilon_ratio)

    def to_inn(self):
        bs = self.linear.bias.data.numpy()
        ws = self.linear.weight.data.numpy()
        self.update_eps()
        return ws, bs, self.ptb_weight.eps, self.ptb_bias.eps

    def forward(self, x):
        return self.linear(x)

    def forward_with_noise(self, x, cfx):
        # only for test use
        layer = self.linear
        weights = layer.weight.data + (
                0.5 - torch.rand(layer.weight.data.shape)) * 2 * self.ptb_weight.eps
        bias = layer.bias.data + (
                0.5 - torch.rand(layer.bias.data.shape)) * 2 * self.ptb_bias.eps
        x = F.linear(x, weights, bias)
        cfx = F.linear(cfx, weights, bias)
        return x, cfx

    def difference(self, other):
        return np.array([np.max(np.abs(self.linear.weight.data.numpy() - other.linear.weight.data.numpy())),
                         np.max(np.abs(self.linear.bias.data.numpy() - other.linear.bias.data.numpy()))])


class LinearBlock(nn.Module):
    def __init__(self, input_dim, out_dim, epsilon_ratio, activation, dropout):
        super().__init__()
        self.linear = BoundedLinear(input_dim, out_dim, epsilon_ratio)
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
    def __init__(self, dims, epsilon_ratio, activation, dropout):
        super().__init__()
        num_blocks = len(dims)
        self.blocks = []
        for i in range(1, num_blocks):
            self.blocks.append(LinearBlock(dims[i - 1], dims[i], epsilon_ratio, activation, dropout=dropout))
        self.model = nn.Sequential(*self.blocks)

    def forward(self, x):
        return self.model(x)

    def forward_with_noise(self, x, cfx_x):
        x = x, cfx_x
        for block in self.blocks:
            x = block.forward_with_noise(*x)
        return x

    def set_eps_ratio(self, eps_ratio):
        for block in self.blocks:
            block.linear.set_eps_ratio(eps_ratio)

    def update_eps(self, eps_ratio=None):
        for block in self.blocks:
            block.linear.update_eps(eps_ratio)

    def difference(self, other):
        ds = []
        for block, block1 in zip(self.blocks, other.blocks):
            ds.append(block.linear.difference(block1.linear))
        ds = np.array(ds)
        print(ds)
        return np.max(ds, axis=0)


class FNN(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens: list, epsilon_ratio=0.0, activation=nn.ReLU,
                 dropout=0):
        super(FNN, self).__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.activation = activation
        self.dropout = dropout
        dims = [num_inputs] + num_hiddens
        self.encoder = MultilayerPerception(dims, epsilon_ratio, activation, dropout=0)  # set dropout to 0
        self.final_fc = BoundedLinear(num_hiddens[-1], num_outputs, epsilon_ratio)
        self.epsilon_ratio = epsilon_ratio

    def forward(self, x):
        x = x.float()  # necessary for Counterfactual generation to work
        x = self.encoder(x)
        x = self.final_fc(x)
        return x

    def forward_separate(self, x):
        # for counternet
        x = x.float()
        h = self.encoder(x)
        x = self.final_fc(h)
        return h, x

    def forward_with_noise(self, x, cfx_x):
        x = x.float(), cfx_x.float()
        x = self.encoder.forward_with_noise(*x)
        x = self.final_fc.forward_with_noise(*x)
        return x

    def difference(self, other):
        d1 = self.encoder.difference(other.encoder)
        d2 = self.final_fc.difference(other.final_fc)
        print(d1, d2)
        return np.maximum(d1, d2)

    def to_Inn(self):
        num_layers = len(self.num_hiddens) + 2  # count input and output layers
        nodes = {}
        nodes[0] = [Node(0, i) for i in range(self.num_inputs)]
        for i in range(1, num_layers - 1):
            nodes[i] = [Node(i, j) for j in range(self.num_hiddens[i - 1])]
        # here the paper assumes the output layer has 1 node, but we have multiple nodes
        nodes[num_layers - 1] = [Node(num_layers - 1, 0)]
        weights = {}
        biases = {}
        for i in range(num_layers - 2):
            ws, bs, epsilon, bias_epsilon = self.encoder.blocks[i].linear.to_inn()
            for node_from in nodes[i]:
                for node_to in nodes[i + 1]:
                    # round by 4 decimals
                    w_val = round(ws[node_to.index][node_from.index], 8)
                    weights[(node_from, node_to)] = Interval(w_val, w_val - epsilon, w_val + epsilon)
                    b_val = round(bs[node_to.index], 8)
                    biases[node_to] = Interval(b_val, b_val - bias_epsilon, b_val + bias_epsilon)

        ws, bs, epsilon, bias_epsilon = self.final_fc.to_inn()
        for node_from in nodes[num_layers - 2]:
            for node_to in nodes[num_layers - 1]:
                # round by 4 decimals
                w_val = round(ws[1][node_from.index] - ws[0][node_from.index], 8)
                weights[(node_from, node_to)] = Interval(w_val, w_val - epsilon * 2, w_val + epsilon * 2)
                b_val = round(bs[1] - bs[0], 8)
                biases[node_to] = Interval(b_val, b_val - bias_epsilon * 2, b_val + bias_epsilon * 2)

        return num_layers, nodes, weights, biases


class EncDec(nn.Module):
    def __init__(self, enc_dims: FNNDims, dec_dims: FNNDims, num_outputs, epsilon_ratio=0.0, activation=nn.ReLU,
                 dropout=0):
        super(EncDec, self).__init__()
        self.activation = activation
        self.dropout = dropout
        self.enc_dims = enc_dims
        self.dec_dims = dec_dims
        self.encoder = MultilayerPerception([enc_dims.in_dim] + enc_dims.hidden_dims, epsilon_ratio, activation,
                                            dropout=dropout)
        self.decoder = MultilayerPerception([dec_dims.in_dim] + dec_dims.hidden_dims, epsilon_ratio, activation,
                                            dropout=dropout)
        self.final_fc = BoundedLinear(dec_dims.hidden_dims[-1], num_outputs, epsilon_ratio)
        self.epsilon_ratio = epsilon_ratio

    def forward(self, x):
        x = x.float()  # necessary for Counterfactual generation to work
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_fc(x)
        return x

    def forward_separate(self, x):
        # for counternet
        x = x.float()
        e = self.encoder(x)
        h = self.decoder(e)
        pred = self.final_fc(h)
        return e, h, pred

    def set_eps_ratio(self, eps_ratio):
        self.encoder.set_eps_ratio(eps_ratio)
        self.decoder.set_eps_ratio(eps_ratio)
        self.final_fc.set_eps_ratio(eps_ratio)

    def update_eps(self, eps_ratio=None):
        self.encoder.update_eps(eps_ratio)
        self.decoder.update_eps(eps_ratio)
        self.final_fc.update_eps(eps_ratio)

    def difference(self, other):
        d1 = self.encoder.difference(other.encoder)
        d2 = self.decoder.difference(other.decoder)
        d3 = self.final_fc.difference(other.final_fc)
        print(d1, d2, d3)
        return np.maximum(np.maximum(d1, d2), d3)

    def to_Inn(self):
        num_layers = 1 + len(self.enc_dims.hidden_dims) + len(
            self.dec_dims.hidden_dims) + 1  # count input and output layers
        nodes = {}
        nodes[0] = [Node(0, i) for i in range(self.enc_dims.in_dim)]
        for i in range(len(self.enc_dims.hidden_dims)):
            nodes[i + 1] = [Node(i + 1, j) for j in range(self.enc_dims.hidden_dims[i])]
        inter_layer_id = len(self.enc_dims.hidden_dims)
        for i in range(len(self.dec_dims.hidden_dims)):
            nodes[inter_layer_id + i + 1] = [Node(inter_layer_id + i + 1, j) for j in
                                             range(self.dec_dims.hidden_dims[i])]
        # here the paper assumes the output layer has 1 node, but we have multiple nodes
        nodes[num_layers - 1] = [Node(num_layers - 1, 0)]
        weights = {}
        biases = {}
        for i in range(len(self.encoder.blocks)):
            ws, bs, epsilon, bias_epsilon = self.encoder.blocks[i].linear.to_inn()
            for node_from in nodes[i]:
                for node_to in nodes[i + 1]:
                    # round by 4 decimals
                    w_val = round(ws[node_to.index][node_from.index], 8)
                    weights[(node_from, node_to)] = Interval(w_val, w_val - epsilon, w_val + epsilon)
                    b_val = round(bs[node_to.index], 8)
                    biases[node_to] = Interval(b_val, b_val - bias_epsilon, b_val + bias_epsilon)

        for j in range(len(self.decoder.blocks)):
            ws, bs, epsilon, bias_epsilon = self.decoder.blocks[j].linear.to_inn()
            i = inter_layer_id + j
            for node_from in nodes[i]:
                for node_to in nodes[i + 1]:
                    # round by 4 decimals
                    w_val = round(ws[node_to.index][node_from.index], 8)
                    weights[(node_from, node_to)] = Interval(w_val, w_val - epsilon, w_val + epsilon)
                    b_val = round(bs[node_to.index], 8)
                    biases[node_to] = Interval(b_val, b_val - bias_epsilon, b_val + bias_epsilon)

        ws, bs, epsilon, bias_epsilon = self.final_fc.to_inn()
        for node_from in nodes[num_layers - 2]:
            for node_to in nodes[num_layers - 1]:
                # round by 4 decimals
                w_val = round(ws[1][node_from.index] - ws[0][node_from.index], 8)
                weights[(node_from, node_to)] = Interval(w_val, w_val - epsilon * 2, w_val + epsilon * 2)
                b_val = round(bs[1] - bs[0], 8)
                biases[node_to] = Interval(b_val, b_val - bias_epsilon * 2, b_val + bias_epsilon * 2)

        return num_layers, nodes, weights, biases


class CounterNet(nn.Module):
    def __init__(self, enc_dims: FNNDims, pred_dims: FNNDims, exp_dims: FNNDims, num_outputs,
                 epsilon_ratio=0.0, activation=nn.ReLU, dropout=0, preprocessor=None,
                 config=None):
        super(CounterNet, self).__init__()
        assert enc_dims.is_before(pred_dims)
        assert enc_dims.is_before(exp_dims)
        exp_dims.in_dim += pred_dims.hidden_dims[-1]  # add the prediction outputs to the explanation
        self.encoder_net_ori = EncDec(enc_dims, pred_dims, num_outputs, epsilon_ratio, activation, 0)
        self.dummy_input_shape = (2, enc_dims.in_dim)
        self.loss_1 = config["loss_1"]
        self.encoder_verify = VerifyModel(self.encoder_net_ori, self.dummy_input_shape, loss_func=self.loss_1)
        self.explainer = nn.Sequential(
            MultilayerPerception([exp_dims.in_dim] + exp_dims.hidden_dims, 0, activation, dropout),
            nn.Linear(exp_dims.hidden_dims[-1], enc_dims.in_dim))
        self.preprocessor = preprocessor  # for normalization
        self.loss_2 = get_loss_by_type(config["loss_2"])
        self.loss_3 = get_loss_by_type(config["loss_3"])
        self.lambda_1 = config["lambda_1"]
        self.lambda_2 = config["lambda_2"]
        self.lambda_3 = config["lambda_3"]

    def forward(self, x, hard=False):
        e, h, pred = self.encoder_net_ori.forward_separate(x)
        e = torch.cat((e, h), -1)
        cfx = self.explainer(e)
        cfx = self.preprocessor.normalize(cfx, hard)
        return cfx, pred

    def forward_point_weights_bias(self, x):
        return self.encoder_net_ori(x)

    def predict(self, x):
        if isinstance(x, list):
            x = np.array(x)
        x = torch.tensor(x)
        return self.encoder_net_ori(x).argmax(dim=-1).numpy()

    def predict_proba(self, x):
        if isinstance(x, list):
            x = np.array(x)
        x = torch.tensor(x)
        return self.encoder_net_ori(x).softmax(dim=-1).detach().numpy()

    def difference(self, other):
        return self.encoder_net_ori.difference(other.encoder_net_ori)

    def get_loss(self, x, y, cfx, y_hat, y_prime_hat, is_cfx, ratio, loss_type="ours"):
        losses = self.get_explainer_loss(x, cfx, y_hat, y_prime_hat)
        return self.get_predictor_loss(x, y) + \
               self.encoder_verify.get_loss_cfx(x, y, cfx, is_cfx, ratio, loss_type) + \
               losses[0] + losses[1]

    def get_predictor_loss(self, x, y):
        return self.encoder_verify.get_loss_ori(x, y) * self.lambda_1

    def get_cfx_loss_pred(self, x, y, cfx, is_cfx, ratio, loss_type="ours"):
        """
        cfx loss on the predictor side, the cfx does not have gradient, thus, not updating the explainer
        """
        return self.encoder_verify.get_loss_cfx(x, y, cfx, is_cfx, ratio, loss_type, correct_pred_only=True)

    def get_cfx_loss_exp(self, x, y, cfx, is_cfx, ratio, loss_type="ours"):
        """
        cfx loss on the explainer side, the cfx has gradient to update the explainer
        """
        return self.encoder_verify.get_loss_cfx(x, y, cfx, is_cfx, ratio, loss_type)

    def get_explainer_loss(self, x, cfx, y_hat, y_prime_hat):  # use y_hat as the ground truth
        return self.loss_2(x.float(), cfx).mean() * self.lambda_2, \
               self.loss_3(y_prime_hat, 1.0 - y_hat).mean() * self.lambda_3

    def save(self, filename):
        torch.save(self.encoder_net_ori.state_dict(), filename + "_encoder_net_ori.pt")
        torch.save(self.explainer.state_dict(), filename + "_explainer.pt")

    def load(self, filename):
        self.encoder_net_ori.load_state_dict(torch.load(filename + "_encoder_net_ori.pt"))
        self.explainer.load_state_dict(torch.load(filename + "_explainer.pt"))
        self.encoder_verify = VerifyModel(self.encoder_net_ori, self.dummy_input_shape, loss_func=self.loss_1)


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
    model = VerifyModel(model_ori, x[2:].shape)
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
