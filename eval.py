import torch
import torch.nn.functional as F
import numpy as np

import argparse

from models.inn import Inn
from models.IBPModel import FNN
from utils import optsolver
from utils import cfx
from utils import dataset

import pickle

'''
Evaluate the robustness of counterfactual explanations
'''


def create_CFX(args, model, minmax, train_data, test_data, num_hiddens):
    @torch.no_grad()
    def predictor(X: np.ndarray) -> np.ndarray:
        X = torch.as_tensor(X, dtype=torch.float32)
        with torch.no_grad():
            ret = model.forward_point_weights_bias(X)
            ret = F.softmax(ret, dim=1)
            ret = ret.cpu().numpy()
        return ret

    cfx_generator = cfx.CFX_Generator(predictor, train_data, num_layers=len(num_hiddens))

    if args.cfx == 'wachter':
        cfx_x, is_cfx = cfx_generator.run_wachter(scaler=minmax, max_iter=500,
                                                  test_instances=test_data.X, lam_init=0.001, max_lam_steps=10)
    else:
        cfx_x, is_cfx = cfx_generator.run_proto(scaler=minmax, theta=10., onehot=args.onehot, test_instances=test_data.X)

    return cfx_x, is_cfx


def load_CFX(args):
    with open(args.CEfile, 'rb') as f:
        cfx_x = pickle.load(f)
    with open(args.isCEfile, 'rb') as f:
        is_cfx = pickle.load(f)
    return cfx_x, is_cfx


def main(args):
    torch.random.manual_seed(0)

    if args.cfx == 'proto':
        feature_types = dataset.CREDIT_FEAT_PROTO
    else:
        feature_types = dataset.CREDIT_FEAT

    if args.onehot:
        minmax = None
        train_data, min_vals, max_vals = dataset.load_data("data/german_train.csv", "credit_risk", feature_types)
        test_data, _, _ = dataset.load_data("data/german_test.csv", "credit_risk", feature_types, min_vals, max_vals)
    else:
        train_data, test_data, minmax = dataset.load_data_v1("data/german_train.csv", "data/german_test.csv",
                                                             "credit_risk",
                                                             feature_types)

    if args.num_to_run is not None:
        test_data.X = test_data.X[:args.num_to_run]
        test_data.y = test_data.y[:args.num_to_run]

    dim_in = train_data.num_features

    num_hiddens = [10, 10]

    model = FNN(dim_in, 2, num_hiddens, epsilon=args.epsilon, bias_epsilon=args.bias_epsilon)
    model.load_state_dict(torch.load(args.model))
    model.eval()
    inn = Inn.from_IBPModel(model)

    if args.CEfile == None:
        cfx_x, is_cfx = create_CFX(args, model, minmax, train_data, test_data, num_hiddens)
    else:
        cfx_x, is_cfx = load_CFX(args)

    pred_y_cor = model.forward_point_weights_bias(torch.tensor(test_data.X).float()).argmax(dim=-1) == torch.tensor(
        test_data.y)

    print(f"Model accuracy: {round(torch.sum(pred_y_cor).item() / len(pred_y_cor) * 100, 2)}% "
          f"({torch.sum(pred_y_cor).item()}/{len(pred_y_cor)})")
    is_cfx = is_cfx & pred_y_cor
    total_valid = torch.sum(is_cfx).item() / len(is_cfx)
    print(f"Found CFX for {round(total_valid * 100, 2)}%"
          f" ({torch.sum(is_cfx).item()}/{len(is_cfx)}) of test points")

    # evaluate robustness first of the model on the test points
    # then evaluate the robustness of the counterfactual explanations

    _, cfx_output = model.get_diffs_binary(torch.tensor(test_data.X).float(), cfx_x,
                                           torch.tensor(test_data.y).bool())
    is_real_cfx = torch.where(torch.tensor(test_data.y) == 0, cfx_output > 0, cfx_output < 0) & is_cfx

    print(f"Of CFX we found, these are robust (by our over-approximation): "
          f"{round(torch.sum(is_real_cfx).item() / torch.sum(is_cfx).item() * 100, 2)}%"
          f" ({torch.sum(is_real_cfx).item()}/{torch.sum(is_cfx).item()})")
    # TODO eventually eval how good (feasible) CFX are

    # TODO figure this out - what ordinal validity checks do we want?
    # before running our checks, if using proto, change data types back to original (include ordinal constraints)
    # but not completely - will not fix ordinal encoding
    # also problematic for training data
    # if args.cfx == 'proto':
    #     test_data.feature_types = dataset.CREDIT_FEAT

    solver_robust_cnt = 0
    solver_bound_better = 0
    for i, (x, y, cfx_x_, is_cfx_, loose_bound) in enumerate(zip(test_data.X, test_data.y, cfx_x, is_cfx, cfx_output)):
        if is_cfx_:
            # print("x: ", x, "y: ", y, "cfx: ", cfx_x_)
            solver = optsolver.OptSolver(test_data, inn, 1 - y, x, mode=1, x_prime=cfx_x_)
            res, bound = solver.compute_inn_bounds()
            # print(i, 1 - y, res, bound, loose_bound.item())
            if abs(bound - loose_bound.item()) > 1e-2:
                solver_bound_better += 1
                # print(i, 1 - y, res, bound, loose_bound.item())
            if res == 1:
                solver_robust_cnt += 1
    print(f"Of CFX we found, these are robust (by the MILP solver): "
          f"{round(solver_robust_cnt / torch.sum(is_cfx).item() * 100, 2)}%"
          f" ({solver_robust_cnt}/{torch.sum(is_cfx).item()})")
    print(f"Of CFX we found, the solver bound is better than the loose bound: "
          f"{round(solver_bound_better / torch.sum(is_cfx).item() * 100, 2)}%"
          f" ({solver_bound_better}/{torch.sum(is_cfx).item()})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='path to model with .pt extension')
    parser.add_argument('--cfx', type=str, default="wachter")
    parser.add_argument('--num_to_run', type=int, default=None, help='number of test examples to run')
    parser.add_argument('--onehot', action='store_true', help='whether to use one-hot encoding')
    parser.add_argument('--CEfile', type=str, default=None, help='path to CE file')
    parser.add_argument('--isCEfile', type=str, default=None, help='path to file specifying whether CE is real or not')

    parser.add_argument('--epsilon', type=float, default=1e-2, help='epsilon for IBP')
    parser.add_argument('--bias_epsilon', type=float, default=1e-1, help='bias epsilon for IBP')
    args = parser.parse_args()

    assert args.cfx in ["wachter", "proto"]

    main(args)
