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
    train_data = dataset.Custom_Dataset("data/german_train.csv", "credit_risk")
    test_data = dataset.Custom_Dataset("data/german_test.csv", "credit_risk")
    minmax = MinMaxScaler(clip=True)
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

    cfx_x, is_cfx = cfx_generator.run_wachter(scaler=minmax)

    # evaluate robustness first of the model on the test points
    # then evaluate the robustness of the counterfactual explanations
    is_real_cfx, cfx_output = model.get_diffs_binary(torch.tensor(test_data.X).float(), cfx_x, torch.tensor(test_data.y).bool())
    # if CFX was fake (i.e., None represented by [0]), then fix is_real_cfx to reflect this
    is_real_cfx = is_real_cfx & is_cfx

    print(sum(is_real_cfx)/len(is_real_cfx))
    # TO DO eventually eval how good (feasible) CFX are
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='path to model with .pt extension')
    args = parser.parse_args()

    main(args)