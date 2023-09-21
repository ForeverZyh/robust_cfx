import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np

from models.inn import Inn
from utils.dataset import Custom_Dataset
from utils import optsolver
from utils.utilities import TOLERANCE


class CFXEvaluator:
    def __init__(self, cfx_x, is_cfx, model_ori, model_shift, train_data: Custom_Dataset,
                 test_data: Custom_Dataset, inn: Inn, log_file):
        self.cfx_x = cfx_x
        self.is_cfx = is_cfx
        self.model_ori = model_ori
        self.model_shift = model_shift
        self.train_data = train_data
        self.test_data = test_data
        self.inn = inn
        self.log_filename = log_file

    def evaluate(self):
        ret = ""
        pred_y_cor = self.model_ori.forward_point_weights_bias(torch.tensor(self.test_data.X).float()).argmax(
            dim=-1) == torch.tensor(self.test_data.y)
        total_cor = torch.sum(pred_y_cor).item()
        pred_y_cor_train = self.model_ori.forward_point_weights_bias(torch.tensor(self.train_data.X).float()).argmax(
            dim=-1) == torch.tensor(self.train_data.y)

        # Accuracy
        ret += f"Model test accuracy: {round(total_cor / len(pred_y_cor) * 100, 2)}%" \
               f" ({torch.sum(pred_y_cor).item()}/{len(pred_y_cor)})\n"
        ret += f"Model train accuracy: {round(torch.sum(pred_y_cor_train).item() / len(pred_y_cor_train) * 100, 2)}% " \
               f"({torch.sum(pred_y_cor_train).item()}/{len(pred_y_cor_train)})\n"

        # Validity
        # TODO figure this out - what ordinal validity checks do we want?
        # before running our checks, if using proto, change data types back to original (include ordinal constraints)
        # but not completely - will not fix ordinal encoding
        # also problematic for training data
        # if args.cfx == 'proto':
        #     test_data.feature_types = dataset.CREDIT_FEAT
        is_cfx = self.is_cfx & pred_y_cor
        total_valid = torch.sum(is_cfx).item()
        ret += f"Validity: {round(total_valid / total_cor * 100, 2)}%" \
               f" ({total_valid}/{total_cor})\n"

        # Robustness
        _, cfx_output = self.model_ori.get_diffs_binary(torch.tensor(self.test_data.X).float(), self.cfx_x,
                                                        torch.tensor(self.test_data.y).bool())
        is_real_cfx = torch.where(torch.tensor(self.test_data.y) == 0, cfx_output > 0, cfx_output < 0) & is_cfx
        ret += f"Robustness (by our over-approximation): " \
               f"{round(torch.sum(is_real_cfx).item() / total_cor * 100, 2)}%" \
               f" ({torch.sum(is_real_cfx).item()}/{total_cor})\n"

        # TODO check what the activation functions of the model is using. If not ReLU, then we skip the following
        solver_robust_cnt = 0
        solver_bound_better = 0
        for i, (x, y, cfx_x_, is_cfx_, loose_bound) in enumerate(
                zip(self.test_data.X, self.test_data.y, self.cfx_x, is_cfx, cfx_output)):
            if is_cfx_:
                solver = optsolver.OptSolver(self.test_data, self.inn, 1 - y, x, mode=1, x_prime=cfx_x_)
                res, bound = solver.compute_inn_bounds()
                if bound is not None and abs(bound - loose_bound.item()) > TOLERANCE:
                    solver_bound_better += 1
                if res == 1:
                    solver_robust_cnt += 1
        ret += f"Robustness (by the MILP solver): " \
               f"{round(solver_robust_cnt / total_cor * 100, 2)}%" \
               f" ({solver_robust_cnt}/{total_cor})\n"
        ret += f"{solver_bound_better} solver bounds are better than ours.\n"

        # Proximity & Sparsity
        proximity = []
        sparsity = []
        for i, (x, cfx_x_, is_cfx_) in enumerate(zip(self.test_data.X, self.cfx_x, is_cfx)):
            if is_cfx_:
                proximity.append(torch.norm(torch.tensor(x) - cfx_x_, p=1).item() / len(x))
                sparsity.append(torch.sum(torch.abs(torch.tensor(x) - cfx_x_) > TOLERANCE).item() / len(x))
        ret += f"Proximity mean: {round(np.mean(proximity), 4)}, std: {round(np.std(proximity), 4)}\n"
        ret += f"Sparsity mean: {round(np.mean(sparsity), 4)}, std: {round(np.std(sparsity), 4)}\n"

        # Distance to data manifold
        dist = []
        neigh = NearestNeighbors(n_neighbors=1, p=1)
        neigh.fit(self.train_data.X)
        for i, (cfx_x_, is_cfx_) in enumerate(zip(self.cfx_x, is_cfx)):
            if is_cfx_:
                dist.append(
                    neigh.kneighbors(np.array([cfx_x_.cpu().numpy()]), return_distance=True)[0][0][0] / len(cfx_x_))
        ret += f"Distance to data manifold mean: {round(np.mean(dist), 4)}, std: {round(np.std(dist), 4)}\n"

        return ret

    def log(self):
        if self.log_filename is not None:
            with open(self.log_filename, "w") as f:
                f.write(self.evaluate())
                f.write("\n")
        else:
            print(self.evaluate())
