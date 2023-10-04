# Adapted from https://github.com/junqi-jiang/robust-ce-inn/blob/main/optsolver.py

from gurobipy import *
from utils.dataset import *
import numpy as np

"""
Class OptSolver can be used to compute counterfactual (mode 0 with delta=0), 
or to compute lower/upper bound of an INN (mode 1 with epsilon not used)  
"""


class OptSolver:
    def __init__(self, dataset, inn, y_prime, x, mode=0, eps=0.0001, M=1000, x_prime=None):
        self.mode = mode  # mode 0: compute counterfactual, mode 1: compute lower/upper bound of INN given a delta
        self.dataset = dataset
        self.inn = inn
        self.y_prime = y_prime  # if 0, constraint: upper output node < 0, if 1, constraint: lower output node >= 0
        self.x = x  # explainee instance x
        self.model = Model()  # initialise Gurobi optimisation model
        self.x_prime = None  # counterfactual instance
        self.eps = eps
        self.M = M
        self.act = inn.act
        if x_prime is not None:
            self.x_prime = np.round(x_prime, 6)
        self.output_node_name = None

    def add_input_variable_constraints(self):
        node_var = dict()
        for feat_idx, feat_mapping in self.dataset.feat_var_map.items():
            # cases by feature type, add different types of variables and constraints
            if self.dataset.feature_types[feat_idx] == DataType.DISCRETE:
                disc_var_list = []
                for var_idx in feat_mapping:
                    if self.mode == 1:
                        node_var[var_idx] = self.model.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS,
                                                              name='x_disc_0_' + str(var_idx))
                    else:
                        node_var[var_idx] = self.model.addVar(vtype=GRB.BINARY, name='x_disc_0_' + str(var_idx))
                    disc_var_list.append(node_var[var_idx])
                    if self.mode == 1:
                        self.model.addConstr(node_var[var_idx] == self.x_prime[var_idx],
                                             name="lbINN_disc_0_" + str(var_idx))
                self.model.update()
                if self.mode == 0:
                    self.model.addConstr(quicksum(disc_var_list) == 1, name='x_disc_0_feat' + str(feat_idx))

            if self.dataset.feature_types[feat_idx] == DataType.ORDINAL:
                prev_var = None
                ord_var_list = []
                for i, var_idx in enumerate(self.dataset.feat_var_map[feat_idx]):
                    if self.mode == 1:
                        node_var[var_idx] = self.model.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS,
                                                              name='x_ord_0_' + str(var_idx))
                    else:
                        node_var[var_idx] = self.model.addVar(vtype=GRB.BINARY, name='x_ord_0_' + str(var_idx))
                    self.model.update()
                    if self.mode == 1:
                        self.model.addConstr(node_var[var_idx] == self.x_prime[var_idx],
                                             name="lbINN_disc_0_" + str(var_idx))
                    if i != 0 and self.mode == 0:
                        self.model.addConstr(prev_var >= node_var[var_idx],
                                             name='x_ord_0_var' + str(var_idx - 1) + '_geq_' + str(var_idx))
                    prev_var = node_var[var_idx]
                    ord_var_list.append(node_var[var_idx])
                if self.mode == 0:
                    self.model.addConstr(quicksum(ord_var_list) >= 1, name='x_ord_0_feat' + str(feat_idx) + '_geq1')

            if self.dataset.feature_types[feat_idx] == DataType.CONTINUOUS_REAL:
                var_idx = self.dataset.feat_var_map[feat_idx][0]
                if self.mode == 1:
                    node_var[var_idx] = self.model.addVar(lb=-float('inf'), ub=1, vtype=GRB.SEMICONT,
                                                          name="x_cont_0_" + str(var_idx))
                else:
                    node_var[var_idx] = self.model.addVar(lb=0, ub=1, vtype=GRB.SEMICONT,
                                                          name="x_cont_0_" + str(var_idx))
                if self.mode == 1:
                    self.model.addConstr(node_var[var_idx] == self.x_prime[var_idx],
                                         name="lbINN_disc_0_" + str(var_idx))
            self.model.update()
        return node_var

    def add_node_variables_constraints(self, node_vars, aux_vars):
        """
        create variables for nodes. Each node has the followings:
        node variable n for the final node value after ReLU,
        auxiliary variable a for the node value before ReLU.

        Constraint on each node:
        n: node variable
        a: binary variable at each node
        M: big value
        n >= 0
        n <= M(1-a)
        n <= ub(W)x + ub(B) + Ma
        n >= lb(W)x + lb(B)
        """
        for i in range(1, self.inn.num_layers):
            node_var = dict()
            aux_var = dict()
            for node in self.inn.nodes[i]:
                self.model.update()
                # hidden layers
                if i != (self.inn.num_layers - 1):
                    node_var[node.index] = self.model.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS,
                                                             name='n_' + str(node))
                    aux_var[node.index] = self.model.addVar(vtype=GRB.BINARY, name='a_' + str(node))
                    self.model.update()
                    lb = quicksum(
                        (self.inn.weights[(node1, node)].lb * node_vars[i - 1][node1.index]) for
                        node1 in self.inn.nodes[i - 1]) + self.inn.biases[node].lb
                    ub = quicksum(
                        (self.inn.weights[(node1, node)].ub * node_vars[i - 1][node1.index]) for
                        node1 in self.inn.nodes[i - 1]) + self.inn.biases[node].ub
                    # constraint 1: node >= 0
                    if self.act == 0:
                        self.model.addConstr(node_var[node.index] >= 0, name="forward_pass_node_" + str(node) + "C1")
                    else:
                        self.model.addConstr(node_var[node.index] >= self.act * lb,
                                             name="forward_pass_node_" + str(node) + "C1")
                    # constraint 2: node <= M(1-a)
                    if self.act == 0:
                        self.model.addConstr(self.M * (1 - aux_var[node.index]) >= node_var[node.index],
                                            name="forward_pass_node_" + str(node) + "C2")
                    else:
                        self.model.addConstr(self.M * (1 - aux_var[node.index]) + self.act * ub >= node_var[node.index],
                                            name="forward_pass_node_" + str(node) + "C2")
                    # constraint 3: node <= ub(W)x + ub(B) + Ma
                    self.model.addConstr(quicksum(
                        (self.inn.weights[(node1, node)].ub * node_vars[i - 1][node1.index]) for
                        node1 in self.inn.nodes[i - 1]) + self.inn.biases[node].ub + self.M * aux_var[node.index] >=
                                         node_var[node.index],
                                         name="forward_pass_node_" + str(node) + "C3")
                    # constraint 4: node >= lb(W)x + lb(B)
                    self.model.addConstr(lb <= node_var[node.index],
                                         name="forward_pass_node_" + str(node) + "C4")
                else:
                    node_var[node.index] = self.model.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS,
                                                             name='n_' + str(node))
                    self.output_node_name = 'n_' + str(node)
                    # constraint 1: node <= ub(W)x + ub(B)
                    self.model.addConstr(quicksum(
                        (self.inn.weights[(node1, node)].ub * node_vars[i - 1][node1.index]) for
                        node1 in self.inn.nodes[i - 1]) + self.inn.biases[node].ub >= node_var[node.index],
                                         name="output_node_pass_" + str(node) + "C1")
                    # constraint 2: node >= lb(W)x + lb(B)
                    self.model.addConstr(quicksum(
                        (self.inn.weights[(node1, node)].lb * node_vars[i - 1][node1.index]) for
                        node1 in self.inn.nodes[i - 1]) + self.inn.biases[node].lb <= node_var[node.index],
                                         name="output_node_pass_" + str(node) + "C2")
                    if self.mode == 1:
                        continue
                    # constraint3: counterfactual constraint for mode 0
                    if self.y_prime:
                        self.model.addConstr(node_var[node.index] - self.eps >= 0.0, name="output_node_lb_>=0")
                    else:
                        self.model.addConstr(node_var[node.index] + self.eps <= 0.0, name="output_node_ub_<0")
                    self.model.update()
            node_vars[i] = node_var
            if i != (self.inn.num_layers - 1):
                aux_vars[i] = aux_var
        return node_vars, aux_vars

    def create_constraints(self):
        node_vars = dict()  # dict of {layer number, {Node's idx int, Gurobi variable obj}}
        aux_vars = dict()  # dict of {layer number, {Node's idx int, Gurobi variable obj}}
        node_vars[0] = self.add_input_variable_constraints()
        node_vars, aux_vars = self.add_node_variables_constraints(node_vars, aux_vars)
        return node_vars, aux_vars

    def set_objective_l1_l0(self, node_vars):
        obj_vars_l1 = []  # each is l1 distance for 1 feature
        # obj_vars_l0 = []  # each is l0 distance for 1 feature
        for feat_idx in range(self.dataset.num_features):
            self.model.update()
            this_obj_var_l1 = self.model.addVar(vtype=GRB.SEMICONT, lb=-GRB.INFINITY, name=f"objl1_feat_{feat_idx}")
            # this_obj_var_l0 = self.model.addVar(vtype=GRB.BINARY, name=f"objl0_feat_{feat_idx}")
            var_idxs = self.dataset.feat_var_map[feat_idx]

            if self.dataset.feature_types[feat_idx] == DataType.DISCRETE:
                to_max = dict()
                for var_idx in var_idxs:
                    to_max[var_idx] = self.model.addVar(vtype=GRB.SEMICONT, lb=-GRB.INFINITY,
                                                        name=f"objl1_feat_disc_{var_idx}")
                    self.model.update()
                    self.model.addConstr(to_max[var_idx] >= (node_vars[0][var_idx] - self.x[var_idx]),
                                         name=f"objl1_feat_disc_{var_idx}_>=-")
                    self.model.addConstr(to_max[var_idx] >= (self.x[var_idx] - node_vars[0][var_idx]),
                                         name=f"objl1_feat_disc_{var_idx}_>=+")  # for abs()
                    self.model.update()
                self.model.addConstr(this_obj_var_l1 == max_([to_max[idx] for idx in to_max.keys()]),
                                     name=f"objl1_feat_{feat_idx}")
                # self.model.addConstr((this_obj_var_l0 == 0) >> (this_obj_var_l1 <= self.eps),
                #                     name=f"objl0_feat_{feat_idx}")  # if l1<= 0, l0=0

            if self.dataset.feature_types[feat_idx] == DataType.ORDINAL:
                self.model.addConstr(this_obj_var_l1 >= (
                        quicksum([node_vars[0][idx] for idx in var_idxs]) - np.sum(self.x[var_idxs])) / (
                                             len(var_idxs) - 1), name=f"objl1_feat_{feat_idx}_>=-")
                self.model.addConstr(this_obj_var_l1 >= (
                        np.sum(self.x[var_idxs]) - quicksum([node_vars[0][idx] for idx in var_idxs])) / (
                                             len(var_idxs) - 1), name=f"objl1_feat_{feat_idx}_>=+")  # for abs()
                # self.model.addConstr((this_obj_var_l0 == 0) >> (this_obj_var_l1 <= self.eps),
                #                     name=f"objl0_feat_{feat_idx}")

            if self.dataset.feature_types[feat_idx] == DataType.CONTINUOUS_REAL:
                self.model.addConstr(
                    this_obj_var_l1 >= (quicksum([node_vars[0][idx] for idx in var_idxs]) - np.sum(self.x[var_idxs])),
                    name=f"objl1_feat_{feat_idx}_>=-")
                self.model.addConstr(
                    this_obj_var_l1 >= (np.sum(self.x[var_idxs]) - quicksum([node_vars[0][idx] for idx in var_idxs])),
                    name=f"objl1_feat_{feat_idx}_>=+")  # for abs()
                # self.model.addConstr((this_obj_var_l0 == 0) >> (this_obj_var_l1 <= self.eps),
                #                     name=f"objl0_feat_{feat_idx}")

            obj_vars_l1.append(this_obj_var_l1)
            # obj_vars_l0.append(this_obj_var_l0)

        # self.model.setObjective(
        #    quicksum(obj_vars_l1) / self.dataset.num_features + quicksum(obj_vars_l0) / self.dataset.num_features,
        #    GRB.MINIMIZE)
        self.model.setObjective(quicksum(obj_vars_l1) / self.dataset.num_features, GRB.MINIMIZE)  # only L1
        self.model.update()

    def set_objective_output_node(self, node_vars):
        if self.y_prime == 1:
            self.model.setObjective(node_vars[self.inn.num_layers - 1][0], GRB.MINIMIZE)
        else:
            self.model.setObjective(node_vars[self.inn.num_layers - 1][0], GRB.MAXIMIZE)

    def compute_counterfactual(self):
        node_vars, aux_vars = self.create_constraints()
        self.set_objective_l1_l0(node_vars)  ## debug
        self.model.Params.LogToConsole = 0  # disable console output
        self.model.optimize()
        xp = []
        try:
            for v in self.model.getVars():
                if 'x_' in v.varName:
                    xp.append(v.getAttr(GRB.Attr.X))
            self.x_prime = np.array(xp)
        except:
            xp = None
        if xp is not None:
            return self.x_prime
        else:
            return None

    def compute_inn_bounds(self):
        """
        test robustness of a counterfactual
        :return: -1: infeasible, 0: not valid, 1: valid
        """
        node_vars, aux_vars = self.create_constraints()
        self.set_objective_output_node(node_vars)
        self.model.Params.LogToConsole = 0  # disable console output
        self.model.optimize()
        res = -1
        bound = None
        try:
            bound = self.model.getVarByName(self.output_node_name).X
            res = 0
            if self.y_prime == 1:
                if bound >= 0:
                    res = 1
            if self.y_prime == 0:
                if bound < 0:
                    res = 1
        except:
            pass
        return res, bound
