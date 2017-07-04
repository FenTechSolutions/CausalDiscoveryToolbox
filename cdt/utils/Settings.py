"""
Settings file for CGNN algorithm
Defining all global variables
Authors : Anonymous Author
Date : 8/05/2017
"""


class Settings(object):
    h_dim = 20
    nb_epoch_train = 700
    nb_epoch_test = 500
    nb_runs = 24
    nb_jobs = 2
    GPU = False
    num_gpu = 1
    gpu_offset = 0
    learning_rate = 0.01
    init_weights = 0.05

    #specific for FSGNN
    nb_run_feature_selection = 1
    regul_param = 0.004
    threshold_UMG = 0.15
    nb_epoch_train_feature_selection = 2000
    nb_epoch_eval_weights = 500


