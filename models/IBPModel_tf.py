import tensorflow as tf
# import numpy as np

# from auto_LiRPA import BoundedModule, BoundedParameter
# from utils.utilities import seed_everything, FAKE_INF, EPS, FNNDims, get_loss_by_type, get_max_loss_by_type


class MultilayerPerception:
    def __init__(self, dims, epsilon_ratio, activation, dropout=0):
        super(MultilayerPerception, self).__init__()
        self.blocks = []
        for i in range(1, len(dims)):
            self.blocks.append(tf.keras.layers.Dense(dims[i]))
            self.blocks.append(activation())

    def build(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class EncDec:
    def __init__(self, enc_dims, dec_dims, num_outputs, epsilon_ratio=0.0, activation=tf.nn.relu, dropout=0):
        self.activation = activation
        self.dropout = dropout
        self.enc_dims = enc_dims
        self.dec_dims = dec_dims
        self.encoder = MultilayerPerception([enc_dims.in_dim] + enc_dims.hidden_dims, epsilon_ratio, activation,
                                            dropout=dropout)
        self.decoder = MultilayerPerception([dec_dims.in_dim] + dec_dims.hidden_dims, epsilon_ratio, activation,
                                            dropout=dropout)
        self.final_fc = tf.keras.layers.Dense(num_outputs, activation=tf.nn.softmax)
        self.epsilon_ratio = epsilon_ratio


class CounterNet:
    def __init__(self, enc_dims, pred_dims, exp_dims, num_outputs,
                 epsilon_ratio=0.0, activation=tf.keras.activations.relu, dropout=0, preprocessor=None,
                 config=None):
        super(CounterNet, self).__init__()
        assert enc_dims.is_before(pred_dims)
        assert enc_dims.is_before(exp_dims)
        exp_dims.in_dim += pred_dims.hidden_dims[-1]  # add the prediction outputs to the explanation
        self.encoder_net_ori = EncDec(enc_dims, pred_dims, num_outputs, epsilon_ratio, activation, 0)
        # self.dummy_input_shape = (2, enc_dims.in_dim)
        # self.loss_1 = config["loss_1"]
        self.encoder_verify = None


    def build(self):
        in_x = tf.keras.Input(shape=(self.encoder_net_ori.enc_dims.in_dim,))
        x = self.encoder_net_ori.encoder.build(in_x)
        x = self.encoder_net_ori.decoder.build(x)
        x = self.encoder_net_ori.final_fc(x)
        self.model = tf.keras.models.Model(inputs=in_x, outputs=x)

    def save(self, filename):
        self.model.save_weights(filename + ".h5")

    def load(self, filename):
        self.model.load_weights(filename + ".h5")
        self.encoder_verify = None
