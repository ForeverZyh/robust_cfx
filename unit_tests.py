import unittest
import torch
import torch.nn.functional as F

from models.IBPModel import FNN
from models.ibp import matmul, IntervalBoundedTensor
TOLERANCE = 1e-2


class TestIBP(unittest.TestCase):
    def test_matmul(self):
        for _ in range(100):
            A = (0.5 - torch.rand(10, 30)) * 2
            B = (0.5 - torch.rand(30, 20)) * 2
            epsilonA = torch.rand(1)
            epsilonB = torch.rand(1)
            A = IntervalBoundedTensor(A, A - epsilonA, A + epsilonA)
            B = IntervalBoundedTensor(B, B - epsilonB, B + epsilonB)
            C = matmul(A, B)
            concrete_A = A.val + (0.5 - torch.rand(A.val.shape)) * 2 * epsilonA
            concrete_B = B.val + (0.5 - torch.rand(B.val.shape)) * 2 * epsilonB
            concrete_C = torch.matmul(concrete_A, concrete_B)
            self.assertTrue(torch.all(C.lb < concrete_C + TOLERANCE))
            self.assertTrue(torch.all(C.ub > concrete_C - TOLERANCE))

    def test_fnn(self):
        eps = 1e-3
        bias_eps = 1e-2
        in_dim = 5
        out_dim = 2
        model = FNN(in_dim, out_dim, [10, 10, 10], epsilon=eps, bias_epsilon=bias_eps)
        batch_size = 10
        with torch.no_grad():
            for _ in range(100):
                x = torch.rand(batch_size, in_dim)
                output = model.forward_except_last(x)
                for i, num_hidden in enumerate(model.num_hiddens):
                    layer = getattr(model, 'fc{}'.format(i))
                    weights = layer.linear.weight + (0.5 - torch.rand(layer.linear.weight.shape)) * 2 * eps
                    bias = layer.linear.bias + (0.5 - torch.rand(layer.linear.bias.shape)) * 2 * bias_eps
                    x = F.linear(x, weights, bias)
                    x = F.relu(x)

                # check whether the ibp bounds are correct
                self.assertTrue(torch.all(output.lb < x + TOLERANCE))
                self.assertTrue(torch.all(output.ub > x - TOLERANCE))

    def test_fnn_diff_tighter(self):
        eps = 1e-2
        bias_eps = 1e-3
        in_dim = 5
        out_dim = 2
        batch_size = 10
        with torch.no_grad():
            for rnd in range(200):
                torch.random.manual_seed(rnd)
                model = FNN(in_dim, out_dim, [2, 3, 4], epsilon=eps, bias_epsilon=bias_eps)
                x = (0.5 - torch.rand(batch_size, in_dim)) * 20
                cfx_x = x + (0.5 - torch.rand(x.shape)) * 10
                output = model.forward(x)
                y = output.val.argmin(dim=-1)
                output_diff = output[:, 0] + (- output[:, 1])
                cfx_output = model.forward(cfx_x)
                cfx_output_diff = cfx_output[:, 0] + (- cfx_output[:, 1])
                # change the sign according to y
                cfx_output_diff = cfx_output_diff * (2 * (0.5 - y))
                output_diff = output_diff * (2 * (0.5 - y))

                is_real_cfx = (output_diff.lb <= 0) & (cfx_output_diff.lb <= 0)
                cfx_output_diff_lb = cfx_output_diff.lb
                # print(is_real_cfx, cfx_output_diff_lb)

                cal_is_real_cfx, cal_cfx_output = model.get_diffs_binary(x, cfx_x, y)
                cal_cfx_output_lb = cal_cfx_output * (2 * (0.5 - y))
                # print(cal_is_real_cfx, cal_cfx_output_lb)
                self.assertTrue(torch.all(cfx_output_diff_lb <= cal_cfx_output_lb + TOLERANCE))
                self.assertTrue(torch.all(is_real_cfx | (~cal_is_real_cfx)))

    def test_fnn_diff_sound(self):
        eps = 1e-2
        bias_eps = 1e-3
        in_dim = 5
        out_dim = 2
        batch_size = 10
        with torch.no_grad():
            for rnd in range(20):
                torch.random.manual_seed(rnd)
                model = FNN(in_dim, out_dim, [10, 3, 4], epsilon=eps, bias_epsilon=bias_eps)
                x_ = (0.5 - torch.rand(batch_size, in_dim)) * 20
                cfx_x_ = x_ + (0.5 - torch.rand(x_.shape)) * 10
                output = model.forward_point_weights_bias(x_)
                y = output.argmin(dim=-1)
                cal_is_real_cfx, cal_cfx_output = model.get_diffs_binary(x_, cfx_x_, y)
                for _ in range(100):
                    x = x_
                    cfx_x = cfx_x_
                    for i, num_hidden in enumerate(model.num_hiddens):
                        layer = getattr(model, 'fc{}'.format(i))
                        weights = layer.linear.weight + (0.5 - torch.rand(layer.linear.weight.shape)) * 2 * eps
                        bias = layer.linear.bias + (0.5 - torch.rand(layer.linear.bias.shape)) * 2 * bias_eps
                        x = F.linear(x, weights, bias)
                        x = F.relu(x)
                        cfx_x = F.linear(cfx_x, weights, bias)
                        cfx_x = F.relu(cfx_x)

                    weights = model.fc_final.linear.weight + (
                            0.5 - torch.rand(model.fc_final.linear.weight.shape)) * 2 * eps
                    bias = model.fc_final.linear.bias + (
                            0.5 - torch.rand(model.fc_final.linear.bias.shape)) * 2 * bias_eps
                    x = F.linear(x, weights, bias)
                    x = x[:, 0] - x[:, 1]
                    cfx_x = F.linear(cfx_x, weights, bias)
                    cfx_x = cfx_x[:, 0] - cfx_x[:, 1]
                    is_real_cfx = torch.where(y == 0, (x <= 0) & (cfx_x <= 0), (x >= 0) & (cfx_x >= 0))
                    is_sound = torch.where(y == 0, cal_cfx_output <= cfx_x + TOLERANCE,
                                           cfx_x - TOLERANCE <= cal_cfx_output)
                    self.assertTrue(torch.all((~is_real_cfx) | cal_is_real_cfx))
                    # print(is_sound, y, cal_cfx_output, cfx_x)
                    self.assertTrue(torch.all(is_sound))
