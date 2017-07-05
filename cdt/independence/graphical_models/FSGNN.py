"""
Feature selection model with generative models
Author : Olivier Goudet

"""
import tensorflow as tf
import os
import pandas as pd
from joblib import Parallel, delayed
from .model import DeconvolutionModel
from cdt.utils.Loss import MMD_loss_tf
from cdt.utils.Graph import *
from cdt.utils.Settings import Settings as SETTINGS
from sklearn.preprocessing import scale


def init(size):
    return tf.random_normal(shape=size, stddev=SETTINGS.init_weights)


def eval_feature_selection_score(df_data, target, verbose=False):
    print("Feature selection for target " + str(target))

    list_features = list(df_data.columns.values)

    N = df_data.shape[0]

    if (target in list_features):
        list_features.remove(target)

    data_features = df_data[list_features]
    data_target = df_data[target]

    data_features = data_features.as_matrix()
    data_target = data_target.as_matrix()

    data_features = data_features.reshape(data_features.shape[0], data_features.shape[1])
    data_target = data_target.reshape(data_target.shape[0], 1)

    n_features = len(list_features)

    all_parent_variables = tf.placeholder(tf.float32, shape=[None, n_features])
    target_variable = tf.placeholder(tf.float32, shape=[None, 1])

    W_in = tf.Variable(init([n_features, SETTINGS.h_dim]))
    W_noise = tf.Variable(init([1, SETTINGS.h_dim]))
    W_input = tf.concat([W_in, W_noise], 0)

    b_in = tf.Variable(init([SETTINGS.h_dim]))
    W_out = tf.Variable(init([SETTINGS.h_dim, 1]))
    b_out = tf.Variable(init([1]))

    input = tf.concat([all_parent_variables, tf.random_normal([N, 1], mean=0, stddev=1)], 1)
    output = tf.nn.relu(tf.matmul(input, W_input) + b_in)
    output = tf.matmul(output, W_out) + b_out

    all_generated_variables = tf.concat([all_parent_variables, output], 1)
    all_real_variables = tf.concat([all_parent_variables, target_variable], 1)

    G_dist_loss = MMD_loss_tf(all_real_variables, all_generated_variables)
    model_complexity = tf.reduce_sum(tf.abs(W_in))
    G_global_loss = G_dist_loss + SETTINGS.regul_param * model_complexity
    G_solver = tf.train.AdamOptimizer(learning_rate=SETTINGS.learning_rate).minimize(G_global_loss)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    avg_weights = 0

    for it in range(SETTINGS.nb_epoch_train_feature_selection):

        _, G_dist_loss_curr, complexity_curr, W_in_curr = sess.run([G_solver, G_dist_loss, model_complexity, W_in],
                                                                   feed_dict={all_parent_variables: data_features,
                                                                              target_variable: data_target})

        if verbose:

            if (it % 100 == 0):

                print('PIter:{}, score:{}, model complexity:{} '.format(it, G_dist_loss_curr, complexity_curr))

                W_in_curr = np.abs(W_in_curr)
                mean_weights = np.mean(W_in_curr, axis=1)
                mean_weights = mean_weights / np.sum(mean_weights)

                maxlist = np.sort(list(mean_weights))[::-1]
                argmaxlist = np.argsort(mean_weights)[::-1]

                for i in range(min(10, n_features)):
                    print(list_features[argmaxlist[i]])
                    print(maxlist[i])

        if (it > SETTINGS.nb_epoch_train_feature_selection - SETTINGS.nb_epoch_eval_weights):
            W_in_curr = np.abs(W_in_curr)
            mean_weights = np.mean(W_in_curr, axis=1)
            mean_weights = mean_weights / np.sum(mean_weights)
            avg_weights += mean_weights

    avg_weights = avg_weights / SETTINGS.nb_epoch_eval_weights

    tf.reset_default_graph()

    return avg_weights


def run_feature_selection(df_data, idx, target):
    if SETTINGS.GPU:
        with tf.device('/gpu:' + str(SETTINGS.gpu_offset + idx % SETTINGS.num_gpu)):
            avg_scores = eval_feature_selection_score(df_data, target)
            return avg_scores
    else:
        avg_scores = eval_feature_selection_score(df_data, target)
        return avg_scores


class FSGNN(DeconvolutionModel):
    def __init__(self):
        super(FSGNN, self).__init__()

    def create_skeleton_from_data(self, data):

        list_nodes = list(data.columns.values)
        n_nodes = len(list_nodes)
        matrix_results = np.zeros((n_nodes, n_nodes))

        data = pd.DataFrame(scale(data), columns=data.columns)

        for _ in range(SETTINGS.nb_run_feature_selection):

            result_feature_selection = Parallel(n_jobs=SETTINGS.nb_jobs)(
                delayed(run_feature_selection)(data, idx, node) for idx, node in enumerate(list_nodes))

            for i in range(len(result_feature_selection)):

                avg_mean = result_feature_selection[i]
                cpt = 0
                for j in range(len(list_nodes)):
                    if (j != i):
                        matrix_results[i, j] = matrix_results[i, j] + avg_mean[cpt]
                        matrix_results[j, i] = matrix_results[j, i] + avg_mean[cpt]
                        cpt += 1

        matrix_results = matrix_results / SETTINGS.nb_run

        if not os.path.exists('results/'):
            os.makedirs('results/')

        graph = UndirectedGraph()

        for i in range(n_nodes):

            for j in range(n_nodes):

                if (j > i):
                    graph.add(data.columns.values[i], data.columns.values[j], matrix_results[i, j])

        return graph
