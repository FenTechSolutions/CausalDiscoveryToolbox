"""
CGNN_graph_model
Author : Olivier Goudet & Diviyan Kalainathan
Ref :
Date : 09/5/17
"""
import tensorflow as tf
import numpy as np
from ...utils.loss import MMD_loss
from ...utils.SETTINGS import CGNN_SETTINGS as SETTINGS
from sklearn.preprocessing import scale


def init(size):
    return tf.random_normal(shape=size, stddev=SETTINGS.init_weights)


class CGNN_graph(object):
    def __init__(self, N, graph, list_nodes, run, pair, learning_rate=SETTINGS.learning_rate):
        """
        Build the tensorflow graph,
        For a given structure
        """
        self.run = run
        self.pair = pair
        n_var = len(list_nodes)

        self.all_real_variables = tf.placeholder(tf.float32, shape=[None, n_var])

        generated_variables = {}
        theta_G = []

        while len(generated_variables) < n_var:
            for var in list_nodes:
                # Check if all parents are generated
                par = graph.get_parents(var)
                if (var not in generated_variables and
                        set(par).issubset(generated_variables)):
                    # Generate the variable
                    W_in = tf.Variable(init([len(par) + 1, SETTINGS.h_dim]))
                    b_in = tf.Variable(init([SETTINGS.h_dim]))
                    W_out = tf.Variable(init([SETTINGS.h_dim, 1]))
                    b_out = tf.Variable(init([1]))

                    input_v = [generated_variables[i] for i in par]
                    input_v.append(tf.random_normal([N, 1], mean=0, stddev=1))
                    input_v = tf.concat(input_v, 1)

                    out_v = tf.nn.relu(tf.matmul(input_v, W_in) + b_in)
                    out_v = tf.matmul(out_v, W_out) + b_out

                    generated_variables[var] = out_v
                    theta_G.extend([W_in, b_in, W_out, b_out])

        listvariablegraph = []
        for var in list_nodes:
            listvariablegraph.append(generated_variables[var])

        all_generated_variables = tf.concat(listvariablegraph, 1)

        self.G_dist_loss_xcausesy = MMD_loss(self.all_real_variables, all_generated_variables)

        # var_list = theta_G
        self.G_solver_xcausesy = (tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.G_dist_loss_xcausesy,
                                                  var_list=theta_G))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def train(self, data, verbose=True):
        for it in range(SETTINGS.nb_epoch_train):

            _, G_dist_loss_xcausesy_curr = self.sess.run(
                [self.G_solver_xcausesy, self.G_dist_loss_xcausesy],
                feed_dict={self.all_real_variables: data}
            )

            if verbose:
                if (it % 100 == 0):
                    print('Pair:{}, Run:{}, Iter:{}, score:{}'.
                          format(self.pair, self.run,
                                 it, G_dist_loss_xcausesy_curr))

    def evaluate(self, data, verbose=True):

        sumMMD_tr = 0

        for it in range(SETTINGS.nb_epoch_test):

            MMD_tr = self.sess.run([self.G_dist_loss_xcausesy], feed_dict={
                self.all_real_variables: data})

            sumMMD_tr += MMD_tr[0]

            if verbose:
                if (it % 100 == 0):
                    print('Pair:{}, Run:{}, Iter:{}, score:{}'
                          .format(self.pair, self.run, it, MMD_tr[0]))

        tf.reset_default_graph()

        return sumMMD_tr / SETTINGS.nb_epoch_test


def run_graph(df_data, graph, idx, run):
    list_nodes = graph.get_list_nodes()
    df_data = df_data[list_nodes].as_matrix()
    data = df_data.astype('float32')

    if SETTINGS.GPU:
        with tf.device('/gpu:' + str(SETTINGS.gpu_offset + run % SETTINGS.num_gpu)):
            model = CGNN_graph(df_data.shape[0], graph, list_nodes, run, idx)
            model.train(data)
            return model.evaluate(data)
    else:
        model = CGNN_graph(df_data.shape[0], graph, list_nodes, run, idx)
        model.train(data)
        return model.evaluate(data)
