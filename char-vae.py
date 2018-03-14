import tensorflow as tf
import numpy as np
from tensorflow.contrib.slim import fully_connected as fc

import matplotlib.pyplot as plt
%matplotlib inline

np.random.seed(0)
tf.set_random_seed(0)

#input diminsions of the model
input_dim = 32
timesteps = 20
batch_size = 64

lstm_size = 128
z_size = 16
num_epochs = 100


class LSTMVAE(object):

    def __init__(self, learning_rate=1e-3):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        #size of the latent 'z' vector
        self.n_z = n_z

        self.build()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def build(self):
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[None,
            timesteps,input_dim)

        #Encoder (x -->LSTM --> z_mean, z_sigma --> z)
        cell_enc = tf.nn.rnn.cell.LSTMCell(lstm_size, forget_bias=1.0)
        e_outputs, e_states = tf.nn.rnn.static_rnn(cell_enc, self.x,
            dtype=tf.float32)

        self.z_mu = fc(e_outputs,z_size, scope='enc_mu', activation_fn=None)
        self.z_log_sigma = fc(e_outputs, z_size, scope = 'enc_log_sigma',
            activation_fn=None)
        eps = tf.random_normal(shape=tf.shape(self.z_log_sigma),mean=0,stddev=1,
            dtype=tf.float32)

        self.z = self.z_mu + tf.sqrt(tf.exp(self.z_log_sigma))*eps

        #Decoder (z --> LSTM --> x_hat)
        cell_dec = tf.nn.rnn.cell.LSTMCell(lstm_size, forget_bias=1.0)
        d_outputs, d_states = tf.nn.rnn.static_rnn(cell_dec, self.z,
            dtype=tf.float32)

        #Loss
        epsilon = 1e-10
        recon_loss = -tf.reduce_sum(self.x * tf.log(epsilon+d_outputs) +
            (1-self.x) * tf.log(epsilon+1-d_outputs), axis = 1)

        self.recon_loss = tf.reduce_mean(recon_loss)

        #Latent Loss

        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma -
            tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq), axis=1)

        self.latent_loss = tf.reduce_mean(latent_loss)

        self.total_loss = tf.reduce_mean(recon_loss + latent_loss)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            .minimize(self.total_loss)

        return
