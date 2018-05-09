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
                 "GPU",  # Number of GPUs Available
                 "default_device",  # Default device for gpu (pytorch 0.4)
                 "autoset_config",
                 "verbose",
                 "r_is_available")

    def __init__(self):
        """Define here the default values of the parameters."""
        super(ConfigSettings, self).__init__()
        self.NB_JOBS = 8
        self.GPU = False
        self.autoset_config = True
        self.verbose = True
        self.default_device = 'cpu'

        if self.autoset_config:
            self = autoset_settings(self)

        self.default_device = 'cuda:0' if self.GPU else 'cpu'

    def __setattr__(self, attr, value):
        """Set attribute override for GPU=True."""
        if attr == "GPU" and value and not self.GPU and self.default_device == 'cpu':
            self.default_device = 'cuda:0'
        super(ConfigSettings, self).__setattr__(attr, value)

    def get_default(self, *args, **kwargs):
        """Get the default parameters as defined in the Settings class."""
        def retrieve_param(i):
            try:
                return self.__getattribute__(i)
            except AttributeError:
                if i == "device":
                    return self.default_device
                else:
                    return self.__getattribute__(i.upper())
        if len(args) == 0:
            if len(kwargs) == 1 and kwargs[list(kwargs.keys())[0]] is not None:
                return kwargs[list(kwargs.keys())[0]]
            elif len(kwargs) == 1:
                return retrieve_param(list(kwargs.keys())[0])
            else:
                raise TypeError("As dict is unordered, it is impossible to give"
                                "the parameters in the correct order.")
        else:
            out = []
            for i in args:
                if i[1] is None:
                    out.append(retrieve_param(i[0]))
                else:
                    out.append(i[1])
            return out


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
            set_var.GPU = len(devices)
            set_var.NB_JOBS = len(devices)

        elif set_var.GPU:
            """2 lines below: Hotfix of incompatibility between
            multiple torch.cuda init through joblib."""
            # if th.cuda.is_available():
            #     set_var.GPU = True
            set_var.GPU = 1
            set_var.NB_JOBS = len(devices)
        else:
            raise KeyError
        print("Detecting CUDA devices : {}".format(devices))

    except KeyError:
        if set_var.GPU:
            warnings.warn("GPU detected but no GPU ID. Setting SETTINGS.GPU to 0")
        else:
            warnings.warn("No GPU automatically detected. Setting SETTINGS.GPU to 0, " +
                          "and SETTINGS.NB_JOBS to cpu_count.")
            set_var.GPU = 0
            set_var.NB_JOBS = multiprocessing.cpu_count()

    return set_var


SETTINGS = ConfigSettings()
