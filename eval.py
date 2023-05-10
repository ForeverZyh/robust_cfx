import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import argparse

from models.IBPModel import FNN
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
    print("Model accuracy: ", round(torch.sum(pred_y_cor).item() / len(pred_y_cor) * 100, 2), "%")
    is_cfx = is_cfx & pred_y_cor
    total_valid = torch.sum(is_cfx).item() / len(is_cfx)
    print("Found CFX for ", round(total_valid * 100, 2), "% of test points")
    # evaluate robustness first of the model on the test points
    # then evaluate the robustness of the counterfactual explanations
    _, cfx_output = model.get_diffs_binary(torch.tensor(test_data.X).float(), cfx_x,
                                           torch.tensor(test_data.y).bool())
    is_real_cfx = torch.where(torch.tensor(test_data.y) == 0, cfx_output > 0, cfx_output < 0)

    print("Of CFX we found, these are robust: ",
          round(torch.sum(is_real_cfx).item() / torch.sum(is_cfx).item() * 100, 2), "%")
    # TO DO eventually eval how good (feasible) CFX are


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='path to model with .pt extension')
    args = parser.parse_args()

    main(args)
