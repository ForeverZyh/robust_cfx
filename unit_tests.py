import unittest
import torch
from torch.nn import ReLU, LeakyReLU, Sigmoid, Tanh

from models.IBPModel import FNN, FAKE_INF, VerifyModel
from utils.utilities import seed_everything, TOLERANCE


class TestIBP(unittest.TestCase):
    def test_fnn_diff_tighter(self):
        eps = 1e-1
        bias_eps = 1e-3
        in_dim = 5
        out_dim = 2
        batch_size = 10
        with torch.no_grad():
            for act in [ReLU, lambda: LeakyReLU(0.1)]:
                print(act)
                tighter_linear_to_crownibp = 0
                tighter_real_linear_to_crownibp = 0
                tighter_crownibp_to_ibp = 0
                tighter_real_crownibp_to_ibp = 0
                for rnd in range(50):
                    seed_everything(rnd)
                    model_ori = FNN(in_dim, out_dim, [10, 10], epsilon_ratio=0.2, activation=act)
                    tmp_x = (0.5 - torch.rand(batch_size * 10, in_dim)) * 20
                    model = VerifyModel(model_ori, tmp_x[:2].shape)
                    tmp_y = model.forward_point_weights_bias(tmp_x).argmax(dim=-1)
                    x = tmp_x[:batch_size]
                    indices = torch.arange(batch_size, batch_size * 2)
                    j = 0
                    for i in range(batch_size * 2, batch_size * 10):
                        if j >= len(indices):
                            break
                        if tmp_y[j] != tmp_y[i]:
                            indices[j] = i
                            j += 1
                    y = tmp_y[:batch_size]
                    cfx_x = tmp_x[indices]
                    cfx_y = tmp_y[indices]

                    # print(y, cfx_y)
                    cal_is_real_cfx_ibp, cal_cfx_output_ibp = \
                        model.get_diffs_binary_ibp(x, cfx_x, y)
                    cal_cfx_output_ibp_lb = cal_cfx_output_ibp * (2 * (0.5 - y))
                    cal_is_real_cfx_crownibp, cal_cfx_output_crownibp = \
                        model.get_diffs_binary_crownibp(x, cfx_x, y)
                    cal_cfx_output_crownibp_lb = cal_cfx_output_crownibp * (2 * (0.5 - y))
                    cal_is_real_cfx, cal_cfx_output = model.get_diffs_binary(x, cfx_x, y)
                    cal_cfx_output_lb = cal_cfx_output * (2 * (0.5 - y))
                    # print(cal_cfx_output_crownibp_lb, cal_cfx_output_ibp_lb)
                    # print(cal_is_real_cfx_crownibp, cal_is_real_cfx)
                    self.assertTrue(torch.all(cal_cfx_output_crownibp_lb <= cal_cfx_output_lb + TOLERANCE))
                    self.assertTrue(torch.all(cal_cfx_output_ibp_lb <= cal_cfx_output_crownibp_lb + TOLERANCE))
                    self.assertTrue(torch.all(cal_is_real_cfx_crownibp | (~cal_is_real_cfx)))
                    self.assertTrue(torch.all(cal_is_real_cfx_ibp | (~cal_is_real_cfx_crownibp)))
                    tighter_linear_to_crownibp += torch.sum(cal_cfx_output_lb - cal_cfx_output_crownibp_lb)
                    tighter_crownibp_to_ibp += torch.sum(cal_cfx_output_crownibp_lb - cal_cfx_output_ibp_lb)
                    tighter_real_linear_to_crownibp += torch.sum(cal_is_real_cfx_crownibp ^ cal_is_real_cfx)
                    tighter_real_crownibp_to_ibp += torch.sum(cal_is_real_cfx_ibp ^ cal_is_real_cfx_crownibp)
                print(act)
                print("tighter_linear_to_crownibp", tighter_linear_to_crownibp.item())
                print("tighter_real_linear_to_crownibp", tighter_real_linear_to_crownibp.item())
                print("tighter_crownibp_to_ibp", tighter_crownibp_to_ibp.item())
                print("tighter_real_crownibp_to_ibp", tighter_real_crownibp_to_ibp.item())

    def test_fnn_diff_sound(self):
        eps = 1e-2
        bias_eps = 1e-3
        in_dim = 5
        out_dim = 2
        batch_size = 10
        with torch.no_grad():
            for act in [ReLU, lambda: LeakyReLU(0.1)]:
                for rnd in range(20):
                    # print(act)
                    seed_everything(rnd)
                    model_ori = FNN(in_dim, out_dim, [3, 4, 5], epsilon_ratio=0.2, activation=act)
                    x_ = (0.5 - torch.rand(batch_size, in_dim)) * 20
                    model = VerifyModel(model_ori, x_[:2].shape)
                    cfx_x_ = x_ + (0.5 - torch.rand(x_.shape)) * 10
                    output = model.forward_point_weights_bias(x_)
                    y = output.argmax(dim=-1)
                    cal_is_real_cfx, cal_cfx_output = model.get_diffs_binary(x_, cfx_x_, y)
                    for _ in range(100):
                        x = x_
                        cfx_x = cfx_x_
                        x, cfx_x = model.ori_model.forward_with_noise(x, cfx_x)
                        x = x[:, 1] - x[:, 0]
                        cfx_x = cfx_x[:, 1] - cfx_x[:, 0]
                        is_real_cfx = torch.where(y == 0, (x <= 0) & (cfx_x <= 0), (x > 0) & (cfx_x > 0))
                        is_sound = torch.where(y == 0, cal_cfx_output <= cfx_x + TOLERANCE,
                                               cfx_x - TOLERANCE <= cal_cfx_output)
                        self.assertTrue(torch.all((~is_real_cfx) | cal_is_real_cfx))
                        # print(is_sound, y, cal_cfx_output, cfx_x)
                        self.assertTrue(torch.all(is_sound))

    def test_get_ub(self):
        eps = 1e-1
        ori_model = FNN(5, 2, [3, 4, 5], epsilon_ratio=0.2)
        model = VerifyModel(ori_model, (2, 5))
        seed_everything(42)
        k = torch.normal(0, 1, (4, 4))
        k = torch.round(k * 100) / 100
        # tensor([[ 1.9300,  1.4900,  0.9000, -2.1100],
        #         [ 0.6800, -1.2300, -0.0400, -1.6000],
        #         [-0.7500,  1.6500, -0.3900, -1.4000],
        #         [-0.7300, -0.5600, -0.7700,  0.7600]])
        b = torch.normal(0, 1, (4,))
        b = torch.round(b * 100) / 100
        b[2] -= 1
        # tensor([0.4600, 0.2700, -0.4700, 0.8100])
        k_1 = torch.normal(0, 1, (4, 4))
        k_1 = torch.round(k_1 * 100) / 100
        # tensor([[ 0.3600, -0.6900, -0.4900,  0.2400],
        #         [-1.1100,  0.0900, -2.3200, -0.2200],
        #         [-0.3100, -0.4000,  0.8000, -0.6200],
        #         [-0.5900, -0.0600, -0.8300,  0.3300]])
        b_1 = torch.normal(0, 1, (4,))
        b_1 = torch.round(b_1 * 100) / 100
        # tensor([ 1.3500,  0.6900, -0.3300,  0.7900])
        w = torch.normal(0, 1, (4,))
        w = torch.round(w * 100) / 100
        # tensor([ 0.0800, -0.1400,  0.3200, -0.4400]) tensor([ 0.4800,  0.2600,  0.7200, -0.0400])
        ub = model.get_ub(k, k_1, b, b_1, w - 2 * eps, w + 2 * eps)
        lb = -model.get_ub(-k, -k_1, -b, -b_1, w - 2 * eps, w + 2 * eps)
        # print(-0.75 * 0.48 + 1.65 * -0.14 - 0.39 * 0.72 - 1.4 * -0.04 - 0.47 + 4 * eps * (1.65 + 0.39))
        # print(0, 0.4 / 1.65, 0.8 / 0.39, 0)
        correct_ub = torch.tensor([0.36 * 0.48 - 0.69 * -0.14 - 0.49 * 0.32 + 0.24 * -0.04 + 1.35,
                                   -1.11 * 0.08 + 0.09 * 0.26 - 2.32 * 0.32 - 0.22 * -0.44 + 0.69,
                                   -0.31 * 0.08 - 0.4 * (0.26 - 0.3548 * eps * 4) + 0.8 * 0.72 - 0.62 * -0.44 - 0.33,
                                   -0.59 * 0.08 + 0.06 * 0.14 - 0.83 * 0.32 - 0.33 * 0.04 + 0.79])  # -0.73 * 0.08 + 0.56 * 0.14 - 0.77 * 0.32 - 0.77 * 0.04 + 0.81 > 0
        correct_lb = torch.tensor([FAKE_INF,  # 1.93 * 0.08 - 0.14 * 1.49 + 0.9 * 0.32 - 2.11 * -0.04 + 0.46 > 0
                                   FAKE_INF,  # 0.68 * 0.08 - 1.23 * 0.26 - 0.04 * 0.72 - 1.6 * -0.04 + 0.27 > 0
                                   -0.31 * 0.48 - 0.4 * 0.26 + 0.8 * 0.32 - 0.62 * -0.04 - 0.33,
                                   -0.59 * 0.48 - 0.06 * 0.26 - 0.83 * 0.72 + 0.33 * -0.44 + 0.79])  # -0.73 * 0.48 - 0.56 * 0.26 - 0.77 * 0.72 + 0.76 * - 0.44 + 0.81 < 0
        # print(1.93 * 0.48 + 1.49 * 0.26 + 0.9 * 0.72 - 2.11 * -0.44 + 0.46 - 4 * eps * (0.9 + 1.49 + 2.11))
        # print(0.68 * 0.48 + 1.23 * 0.14 - 0.04 * 0.32 + 1.6 * 0.44 + 0.27 - 4 * eps * (0.68 + 1.23))
        # print(correct_lb, correct_ub)
        # print(lb, ub)
        self.assertTrue(torch.all((torch.abs(ub - correct_ub) < TOLERANCE) |
                                  (torch.isinf(correct_ub) & torch.isinf(ub))))
        self.assertTrue(torch.all((torch.abs(lb - correct_lb) < TOLERANCE) |
                                  (torch.isinf(correct_lb) & torch.isinf(lb))))
        # the manual test in the document
        self.assertTrue(model.get_ub(torch.tensor([1, 1]), torch.tensor([-1, -1]), torch.tensor([0]), torch.tensor([0]),
                                     torch.tensor([-2 * eps, -2 * eps]), torch.tensor([2 * eps, 2 * eps])).item() == 0)
