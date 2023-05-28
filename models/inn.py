from models import IBPModel


class Interval:
    def __init__(self, value, lb=0, ub=0):
        self.value = value
        self.lb = lb
        self.ub = ub

    def set_bounds(self, lb, ub):
        self.lb = lb
        self.ub = ub

    def get_bound(self, y_prime):
        return self.lb if y_prime is 1 else self.ub


class Node(Interval):
    def __init__(self, layer, index, lb=0, ub=0):
        super().__init__(lb, ub)
        self.layer = layer
        self.index = index
        self.loc = (layer, index)

    def __str__(self):
        return str(self.loc)


class Inn:
    """
    args:
    delta: weight shift value
    nodes: dict of {int layer num, [Node1, Node2, ...]}
    weights: dict of {(Node in prev layer, Node in this layer), Interval}.
    biases: dict of {Node, Interval}
    """

    def __init__(self, num_layers, delta, bias_delta, nodes, weights, biases):
        self.num_layers = num_layers
        self.delta = delta
        self.delta = bias_delta
        self.nodes = nodes
        self.weights = weights
        self.biases = biases

    @classmethod
    def from_IBPModel(cls, model: IBPModel):
        num_layers = len(model.num_hiddens) + 2  # count input and output layers
        delta = model.epsilon
        bias_delta = model.bias_epsilon
        nodes = {}
        nodes[0] = [Node(0, i) for i in range(model.num_inputs)]
        for i in range(1, num_layers - 1):
            nodes[i] = [Node(i, j) for j in range(model.num_hiddens[i - 1])]
        # here the paper assumes the output layer has 1 node, but we have multiple nodes
        nodes[num_layers - 1] = [Node(num_layers - 1, 0)]
        weights = {}
        biases = {}
        for i in range(num_layers - 2):
            layer = getattr(model, 'fc{}'.format(i)).linear
            ws = layer.weight.data.numpy()
            bs = layer.bias.data.numpy()
            for node_from in nodes[i]:
                for node_to in nodes[i + 1]:
                    # round by 4 decimals
                    w_val = round(ws[node_to.index][node_from.index], 8)
                    weights[(node_from, node_to)] = Interval(w_val, w_val - delta, w_val + delta)
                    b_val = round(bs[node_to.index], 8)
                    biases[node_to] = Interval(b_val, b_val - bias_delta, b_val + bias_delta)

        layer = getattr(model, 'fc_final').linear
        ws = layer.weight.data.numpy()
        bs = layer.bias.data.numpy()
        for node_from in nodes[num_layers - 2]:
            for node_to in nodes[num_layers - 1]:
                # round by 4 decimals
                w_val = round(ws[1][node_from.index] - ws[0][node_from.index], 8)
                weights[(node_from, node_to)] = Interval(w_val, w_val - delta * 2, w_val + delta * 2)
                b_val = round(bs[1] - bs[0], 8)
                biases[node_to] = Interval(b_val, b_val - bias_delta * 2, b_val + bias_delta * 2)

        return cls(num_layers, delta, bias_delta, nodes, weights, biases)
