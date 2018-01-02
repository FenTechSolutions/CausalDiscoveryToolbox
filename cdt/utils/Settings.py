"""
Settings file for CGNN algorithm
Defining all global variables
Authors : Anonymous Author
Date : 8/05/2017
"""
import ast
import os
import warnings
import multiprocessing


class ConfigSettings(object):
    __slots__ = ("NB_JOBS",  # Number of parallel jobs runnable
                 "GPU",  # True if GPU is available
                 "NB_GPU",  # Number of available GPUs (Not used if autoset)
                 "GPU_OFFSET",  # First index of GPU (Not used if autoset)
                 "GPU_LIST",  # List of CUDA_VISIBLE_DEVICES
                 "autoset_config",
                 "verbose",
                 "r_is_available",
                 "torch",
                 "threshold_UMG",)

    def __init__(self):  # Define here the default values of the parameters
        super(ConfigSettings, self).__init__()
        self.NB_JOBS = 8
        self.GPU = True
        self.NB_GPU = 4
        self.GPU_OFFSET = 0
        self.GPU_LIST = [i for i in range(
            self.GPU_OFFSET, self.GPU_OFFSET + self.NB_GPU)]
        self.autoset_config = True
        self.verbose = True
        self.r_is_available = False
        self.threshold_UMG = 0.15

        try:
            import torch
            from torch.autograd import Variable
            # Remaining package install only reserve namespace
            self.torch = torch
        except ImportError as e:
            warnings.warn("Torch not available : {}".format(e))
            self.torch = None
        if self.autoset_config:
            self = autoset_settings(self)


class CGNNSettings(object):
    __slots__ = ("h_layer_dim",
                 "train_epochs",
                 "test_epochs",
                 "NB_RUNS",
                 "NB_MAX_RUNS",
                 "learning_rate",
                 "init_weights",
                 "ttest_threshold",
                 "nb_max_loop",
                 "kernel",
                 "nb_run_feature_selection",
                 "regul_param",
                 "nb_epoch_train_feature_selection",
                 "nb_epoch_eval_weights",
                 "use_Fast_MMD",
                 "nb_vectors_approx_MMD",
                 "complexity_graph_param",
                 "max_nb_points",
                 "max_parents_block",
                 "asymmetry_param")

    def __init__(self):
        super(CGNNSettings, self).__init__()
        self.NB_RUNS = 8
        self.NB_MAX_RUNS = 32
        self.learning_rate = 0.01
        self.init_weights = 0.05
        self.max_nb_points = 1500
        self.h_layer_dim = 20
        self.use_Fast_MMD = False
        self.nb_vectors_approx_MMD = 100

        self.train_epochs = 1000
        self.test_epochs = 500
        self.complexity_graph_param = 0.00005
        self.ttest_threshold = 0.01
        self.nb_max_loop = 3

        self.kernel = "RBF"

        # specific for FSGNN
        self.nb_run_feature_selection = 1
        self.nb_epoch_train_feature_selection = 2000
        self.nb_epoch_eval_weights = 500
        self.regul_param = 0.004

        # specific for blockwise CGNN
        self.max_parents_block = 5
        self.asymmetry_param = 0.0001


def autoset_settings(set_var):
    """
    Autoset GPU parameters using CUDA_VISIBLE_DEVICES variables.
    Return default config if variable not set.
    :param set_var: Variable to set. Must be of type ConfigSettings
    """
    try:
        devices = ast.literal_eval(os.environ["CUDA_VISIBLE_DEVICES"])
        if type(devices) != list and type(devices) != tuple:
            devices = [devices]
        if len(devices) != 0:
            set_var.GPU_LIST = devices
            set_var.GPU = True
            set_var.NB_JOBS = len(devices)
            print("Detecting CUDA devices : {}".format(devices))
        else:
            raise KeyError
    except KeyError:
        warnings.warn(
            "No GPU automatically detected. Set SETTINGS.GPU to false," +
            "SETTINGS.GPU_LIST to [], and SETTINGS.NB_JOBS to cpu_count.")
        set_var.GPU = False
        set_var.GPU_LIST = []
        set_var.NB_JOBS = multiprocessing.cpu_count()

    return set_var


SETTINGS = ConfigSettings()
CGNN_SETTINGS = CGNNSettings()
