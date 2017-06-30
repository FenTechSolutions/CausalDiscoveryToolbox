"""
Settings file for CGNN algorithm
Defining all global variables
Authors : Anonymous Author
Date : 8/05/2017
"""


class CGNN_SETTINGS(object):
    h_dim = 20
    nb_epoch_train = 400
    nb_epoch_test = 500
    nb_run = 24
    nb_jobs = 2
    GPU = False
    num_gpu = 1
    gpu_offset = 0
    learning_rate = 0.01
    init_weights = 0.05
    threshold_UMG = 0.0
