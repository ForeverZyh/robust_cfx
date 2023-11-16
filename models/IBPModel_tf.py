import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

from auto_LiRPA import BoundedModule, BoundedParameter
from utils.utilities import seed_everything, FAKE_INF, EPS, FNNDims, get_loss_by_type, get_max_loss_by_type


class MultilayerPerception(tf.keras.layers.Layer):
    def __init__(self, dims, epsilon_ratio, activation, dropout=0):
        super(MultilayerPerception, self).__init__()
        self.blocks = []
        for i in range(len(dims) - 1):
            self.blocks.append(BoundedLinear(dims[i], dims[i + 1], epsilon_ratio))
            if i < len(dims) - 2:
                self.blocks.append(layers.Dropout(dropout))
                self.blocks.append(activation())

    def call(self, x):
        for layer in self.blocks:
            x = layer(x)
        return x

class BoundedLinear(layers.Layer):
    def __init__(self, in_dim, out_dim, epsilon_ratio):
        super(BoundedLinear, self).__init__()
        self.linear = layers.Dense(out_dim)
        self.epsilon_ratio = epsilon_ratio

    def call(self, x):
        return self.linear(x)

    def forward(self, x):
        return self.linear(x)
    
    def to_inn(self):
        bs = self.linear.bias.numpy()
        ws = self.linear.weight.numpy()
        self.update_eps()
        return ws, bs, self.epsilon_ratio, self.epsilon_ratio
        
class Node:
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index

class Interval:
    def __init__(self, center, lower, upper):
        self.center = center
        self.lower = lower
        self.upper = upper

class EncDec(tf.keras.Model):
    def __init__(self, enc_dims, dec_dims, num_outputs, epsilon_ratio=0.0, activation=tf.nn.relu, dropout=0):
        super(EncDec, self).__init__()
        self.activation = activation
        self.dropout = dropout
        self.enc_dims = enc_dims
        self.dec_dims = dec_dims
        self.encoder = MultilayerPerception([enc_dims.in_dim] + enc_dims.hidden_dims, epsilon_ratio, activation,
                                            dropout=dropout)
        self.decoder = MultilayerPerception([dec_dims.in_dim] + dec_dims.hidden_dims, epsilon_ratio, activation,
                                            dropout=dropout)
        self.final_fc = BoundedLinear(dec_dims.hidden_dims[-1], num_outputs, epsilon_ratio)
        self.epsilon_ratio = epsilon_ratio

    def call(self, x):
        x = tf.dtypes.cast(x, dtype=tf.float32)  # necessary for Counterfactual generation to work
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_fc(x)
        return x

    def forward_separate(self, x):
        # for counternet
        x = tf.dtypes.cast(x, dtype=tf.float32)
        e = self.encoder(x)
        h = self.decoder(e)
        pred = self.final_fc(h)
        return e, h, pred

    def set_eps_ratio(self, eps_ratio):
        self.encoder.set_eps_ratio(eps_ratio)
        self.decoder.set_eps_ratio(eps_ratio)
        self.final_fc.set_eps_ratio(eps_ratio)

    def update_eps(self, eps_ratio=None):
        self.encoder.update_eps(eps_ratio)
        self.decoder.update_eps(eps_ratio)
        self.final_fc.update_eps(eps_ratio)

    def difference(self, other):
        d1 = self.encoder.difference(other.encoder)
        d2 = self.decoder.difference(other.decoder)
        d3 = self.final_fc.difference(other.final_fc)
        print(d1, d2, d3)
        return np.maximum(np.maximum(d1, d2), d3)

    def to_inn(self):
        num_layers = 1 + len(self.enc_dims.hidden_dims) + len(
            self.dec_dims.hidden_dims) + 1  # count input and output layers
        nodes = {}
        nodes[0] = [Node(0, i) for i in range(self.enc_dims.in_dim)]
        for i in range(len(self.enc_dims.hidden_dims)):
            nodes[i + 1] = [Node(i + 1, j) for j in range(self.enc_dims.hidden_dims[i])]
        inter_layer_id = len(self.enc_dims.hidden_dims)
        for i in range(len(self.dec_dims.hidden_dims)):
            nodes[inter_layer_id + i + 1] = [Node(inter_layer_id + i + 1, j) for j in
                                             range(self.dec_dims.hidden_dims[i])]
        # here the paper assumes the output layer has 1 node, but we have multiple nodes
        nodes[num_layers - 1] = [Node(num_layers - 1, 0)]
        weights = {}
        biases = {}
        for i in range(len(self.encoder.blocks)):
            ws, bs, epsilon, bias_epsilon = self.encoder.blocks[i].linear.to_inn()
            for node_from in nodes[i]:
                for node_to in nodes[i + 1]:
                    # round by 4 decimals
                    w_val = round(ws[node_to.node_index][node_from.node_index], 8)
                    weights[(node_from, node_to)] = Interval(w_val, w_val - epsilon, w_val + epsilon)
                    b_val = round(bs[node_to.node_index], 8)
                    biases[node_to] = Interval(b_val, b_val - bias_epsilon, b_val + bias_epsilon)

        for j in range(len(self.decoder.blocks)):
            ws, bs, epsilon, bias_epsilon = self.decoder.blocks[j].linear.to_inn()
            i = inter_layer_id + j
            for node_from in nodes[i]:
                for node_to in nodes[i + 1]:
                    # round by 4 decimals
                    w_val = round(ws[node_to.node_index][node_from.node_index], 8)
                    weights[(node_from, node_to)] = Interval(w_val, w_val - epsilon, w_val + epsilon)
                    b_val = round(bs[node_to.node_index], 8)
                    biases[node_to] = Interval(b_val, b_val - bias_epsilon, b_val + bias_epsilon)

        ws, bs, epsilon, bias_epsilon = self.final_fc.to_inn()
        for node_from in nodes[num_layers - 2]:
            for node_to in nodes[num_layers - 1]:
                # round by 4 decimals
                w_val = round(ws[1][node_from.node_index] - ws[0][node_from.node_index], 8)
                weights[(node_from, node_to)] = Interval(w_val, w_val - epsilon * 2, w_val + epsilon * 2)
                b_val = round(bs[1] - bs[0], 8)
                biases[node_to] = Interval(b_val, b_val - bias_epsilon * 2, b_val + bias_epsilon * 2)

        return num_layers, nodes, weights, biases


class LinearBlock(tf.keras.layers.Layer):
    def __init__(self, input_dim, out_dim, epsilon_ratio, activation, dropout):
        super(LinearBlock, self).__init__()
        self.linear = BoundedLinear(input_dim, out_dim, epsilon_ratio)
        if dropout == 0:
            self.block = activation()
        else:
            self.block = tf.keras.Sequential([
                activation(),
                layers.Dropout(dropout),
            ])

    def call(self, x):
        x = self.linear(x)
        return self.block(x)

    def forward_with_noise(self, x, cfx):
        x, cfx = self.linear.forward_with_noise(x, cfx)
        return self.block(x), self.block(cfx)

class MultilayerPerception(tf.keras.layers.Layer):

    def __init__(self, dims, epsilon_ratio, activation, dropout):
        super(MultilayerPerception, self).__init__()
        num_blocks = len(dims)
        self.blocks = []
        for i in range(1, num_blocks):
            self.blocks.append(LinearBlock(dims[i - 1], dims[i], epsilon_ratio, activation, dropout=dropout))
        self.model = tf.keras.Sequential(self.blocks)

    def call(self, x):
        return self.model(x)

    def forward_with_noise(self, x, cfx_x):
        x = (x, cfx_x)
        for block in self.blocks:
            x = block.forward_with_noise(*x)
        return x

    def set_eps_ratio(self, eps_ratio):
        for block in self.blocks:
            block.linear.set_eps_ratio(eps_ratio)

    def update_eps(self, eps_ratio=None):
        for block in self.blocks:
            block.linear.update_eps(eps_ratio)

    def difference(self, other):
        ds = []
        for block, block1 in zip(self.blocks, other.blocks):
            ds.append(block.linear.difference(block1.linear))
        ds = np.array(ds)
        print(ds)
        return np.max(ds, axis=0)
    

class FNN(tf.keras.Model):
    def __init__(self, num_inputs, num_outputs, num_hiddens, epsilon_ratio=0.0, activation=tf.keras.activations.relu,
                 dropout=0):
        super(FNN, self).__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.activation = activation
        self.dropout = dropout
        dims = [num_inputs] + num_hiddens
        self.encoder = MultilayerPerception(dims, epsilon_ratio, activation, dropout=0)  # set dropout to 0
        self.final_fc = BoundedLinear(num_hiddens[-1], num_outputs, epsilon_ratio)
        self.epsilon_ratio = epsilon_ratio

    def call(self, x):
        x = tf.dtypes.cast(x, dtype=tf.float32)  # necessary for Counterfactual generation to work
        x = self.encoder(x)
        x = self.final_fc(x)
        return x

    def forward_separate(self, x):
        # for counternet
        x = tf.dtypes.cast(x, dtype=tf.float32)
        h = self.encoder(x)
        x = self.final_fc(h)
        return h, x

    def forward_with_noise(self, x, cfx_x):
        x = tf.dtypes.cast(x, dtype=tf.float32), tf.dtypes.cast(cfx_x, dtype=tf.float32)
        x = self.encoder.forward_with_noise(*x)
        x = self.final_fc.forward_with_noise(*x)
        return x

    def difference(self, other):
        d1 = self.encoder.difference(other.encoder)
        d2 = self.final_fc.difference(other.final_fc)
        print(d1, d2)
        return tf.math.maximum(d1, d2)
    

class EncDec(tf.keras.Model):
    def __init__(self, enc_dims, dec_dims, num_outputs, epsilon_ratio=0.0, activation=tf.keras.activations.relu,
                 dropout=0):
        super(EncDec, self).__init__()
        self.activation = activation
        self.dropout = dropout
        self.enc_dims = enc_dims
        self.dec_dims = dec_dims
        self.encoder = MultilayerPerception([enc_dims.in_dim] + enc_dims.hidden_dims, epsilon_ratio, activation,
                                            dropout=dropout)
        self.decoder = MultilayerPerception([dec_dims.in_dim] + dec_dims.hidden_dims, epsilon_ratio, activation,
                                            dropout=dropout)
        self.final_fc = BoundedLinear(dec_dims.hidden_dims[-1], num_outputs, epsilon_ratio)
        self.epsilon_ratio = epsilon_ratio

    def call(self, x):
        x = tf.dtypes.cast(x, dtype=tf.float32)  # necessary for Counterfactual generation to work
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_fc(x)
        return x

    def forward_separate(self, x):
        # for counternet
        x = tf.dtypes.cast(x, dtype=tf.float32)
        e = self.encoder(x)
        h = self.decoder(e)
        pred = self.final_fc(h)
        return e, h, pred
    

class CounterNet(tf.keras.Model):
    def __init__(self, enc_dims, pred_dims, exp_dims, num_outputs,
                 epsilon_ratio=0.0, activation=tf.keras.activations.relu, dropout=0, preprocessor=None,
                 config=None):
        super(CounterNet, self).__init__()
        assert enc_dims.is_before(pred_dims)
        assert enc_dims.is_before(exp_dims)
        exp_dims.in_dim += pred_dims.hidden_dims[-1]  # add the prediction outputs to the explanation
        self.encoder_net_ori = EncDec(enc_dims, pred_dims, num_outputs, epsilon_ratio, activation, 0)
        self.dummy_input_shape = (2, enc_dims.in_dim)
        self.loss_1 = config["loss_1"]
        self.encoder_verify = None # changing to none to avoid auto lirpa.. unsure if this'll owrk
                #VerifyModel(self.encoder_net_ori, self.dummy_input_shape, loss_func=self.loss_1)
        self.explainer = tf.keras.Sequential([
            MultilayerPerception([exp_dims.in_dim] + exp_dims.hidden_dims, 0, activation, dropout),
            layers.Dense(enc_dims.in_dim)
        ])
        self.preprocessor = preprocessor  # for normalization
        self.loss_2 = get_loss_by_type(config["loss_2"])
        self.loss_3 = get_loss_by_type(config["loss_3"])
        self.lambda_1 = config["lambda_1"]
        self.lambda_2 = config["lambda_2"]
        self.lambda_3 = config["lambda_3"]

    def call(self, x, hard=False):
        e, h, pred = self.encoder_net_ori.forward_separate(x)
        e = tf.concat([e, h], axis=-1)
        cfx = self.explainer(e)
       #cfx = self.preprocessor.normalize(cfx, hard)
        return cfx, pred

    def forward_point_weights_bias(self, x):
        return self.encoder_net_ori(x)

    def predict(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        return tf.argmax(self.encoder_net_ori(x), axis=-1).numpy()

    def predict_proba(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        return tf.nn.softmax(self.encoder_net_ori(x)).numpy()

    def predict_logits(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        ret = self.encoder_net_ori(x).numpy()
        return ret[:, 1] - ret[:, 0]

    def difference(self, other):
        return self.encoder_net_ori.difference(other.encoder_net_ori)

    def save(self, filename):
        print("HERE")
        self.encoder_net_ori.save(filename + "_encoder_net_ori")
        self.explainer.save(filename + "_explainer")

    def load(self, filename):
        self.encoder_net_ori.load(filename + "_encoder_net_ori")
        self.explainer.load(filename + "_explainer")
        self.encoder_verify = None # odel(self.encoder_net_ori, self.dummy_input_shape, loss_func=self.loss_1)
