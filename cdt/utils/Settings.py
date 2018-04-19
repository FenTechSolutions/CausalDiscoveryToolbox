"""Settings file for the Causal Discovery Toolbox.

Defining all global variables
Author : Diviyan Kalainathan
Date : 8/05/2017
"""
import ast
import os
import warnings
import multiprocessing
import torch as th


def message_warning(msg, *a, **kwargs):
    """Ignore everything except the message."""
    return str(msg) + '\n'


warnings.formatwarning = message_warning


class ConfigSettings(object):
    """Defining the class for the hardware/high level settings of the CDT."""

    __slots__ = ("NB_JOBS",  # Number of parallel jobs runnable
                 "GPU",  # True if GPU is available
                 "GPU_LIST",  # List of CUDA_VISIBLE_DEVICES
                 "autoset_config",
                 "verbose",
                 "r_is_available")

    def __init__(self):
        """Define here the default values of the parameters."""
        super(ConfigSettings, self).__init__()
        self.NB_JOBS = 8
        self.GPU = False
        self.GPU_LIST = []
        self.autoset_config = True
        self.verbose = True

        if self.autoset_config:
            self = autoset_settings(self)


def autoset_settings(set_var):
    """Autoset GPU parameters using CUDA_VISIBLE_DEVICES variables.

    Return default config if variable not set.
    :param set_var: Variable to set. Must be of type ConfigSettings
    """
    try:
        devices = ast.literal_eval(os.environ["CUDA_VISIBLE_DEVICES"])
        if type(devices) != list and type(devices) != tuple:
            devices = [devices]
        if len(devices) != 0:
            set_var.GPU = True
            set_var.GPU_LIST = list(range(len(devices)))
            set_var.NB_JOBS = len(devices)

        elif set_var.GPU:
            """2 lines below: Hotfix of incompatibility between
            multiple torch.cuda init through joblib."""
            # if th.cuda.is_available():
            #     set_var.GPU = True
            set_var.GPU_LIST = [0]
            set_var.NB_JOBS = len(devices)
        else:
            raise KeyError
        print("Detecting CUDA devices : {}".format(devices))

    except KeyError:
        if set_var.GPU:
            warnings.warn("GPU detected but no GPU ID. Setting SETTINGS.GPU_LIST to [0]")
        else:
            warnings.warn("No GPU automatically detected. Setting SETTINGS.GPU to False, " +
                          "SETTINGS.GPU_LIST to [], and SETTINGS.NB_JOBS to cpu_count.")
            set_var.GPU = False
            set_var.GPU_LIST = []
            set_var.NB_JOBS = multiprocessing.cpu_count()

    return set_var


SETTINGS = ConfigSettings()
