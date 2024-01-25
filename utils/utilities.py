import random
import os
import numpy as np
import torch
import torch.nn as nn

TOLERANCE = 1e-2
FAKE_INF = 10
EPS = 1e-8


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FNNDims:
    def __init__(self, in_dim, hidden_dims: list):
        # A fully connected network with input dimension in_dim and hidden dimensions hidden_dims
        if in_dim is None:
            self.in_dim = hidden_dims[0]
            self.hidden_dims = hidden_dims[1:]
        else:
            self.in_dim = in_dim
            self.hidden_dims = hidden_dims

    def is_before(self, other):
        return self.hidden_dims[-1] == other.in_dim


def get_loss_by_type(loss_func, weight = None):
    if loss_func == "bce":
        if weight is None:
            return nn.BCELoss(reduction="none")
        return nn.BCELoss(weight=torch.tensor(weight), reduction="none")
    elif loss_func == "mse":
        return nn.MSELoss(reduction="none")
    else:
        raise NotImplementedError


def get_max_loss_by_type(loss_func):
    if loss_func == "bce":
        return 100.0
    elif loss_func == "mse":
        return 1.0
    else:
        raise NotImplementedError
