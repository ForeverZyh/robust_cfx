import torch
import torch.nn.functional as F
import numpy as np
import tqdm

import argparse

from models.inn import Inn
from models.IBPModel import FNN
from utils import optsolver
from utils import cfx
from utils import dataset

'''
Evaluate the robustness of counterfactual explanations
'''


def main(args):
    torch.random.manual_seed(0)
    train_data, test_data, minmax = dataset.load_data_v1("data/german_train.csv", "data/german_test.csv", "credit_risk",
                                                         dataset.CREDIT_FEAT)
    train_data.X = minmax.fit_transform(train_data.X)
    test_data.X = minmax.transform(test_data.X)
    dim_in = train_data.num_features

    num_hiddens = [10, 10]
    model = FNN(dim_in, 2, num_hiddens, epsilon=1e-2, bias_epsilon=1e-1)
    model.load_state_dict(torch.load(args.model))
    model.eval()
    inn = Inn.from_IBPModel(model)

    @torch.no_grad()
    def predictor(X: np.ndarray) -> np.ndarray:
        X = torch.as_tensor(X, dtype=torch.float32)
        with torch.no_grad():
            ret = model.forward_point_weights_bias(X)
            ret = F.softmax(ret, dim=1)
            ret = ret.cpu().numpy()
            # print(ret)
        return ret

    cfx_generator = cfx.CFX_Generator(predictor, test_data, num_layers=len(num_hiddens))

    cfx_x, is_cfx = cfx_generator.run_wachter(scaler=minmax, max_iter=100)
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

    solver_robust_cnt = 0
    solver_bound_better = 0
    for i, (x, y, cfx_x_, is_cfx_, loose_bound) in enumerate(zip(test_data.X, test_data.y, cfx_x, is_cfx, cfx_output)):
        if is_cfx_:
            # print("x: ", x, "y: ", y, "cfx: ", cfx_x)
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
    args = parser.parse_args()

    main(args)
