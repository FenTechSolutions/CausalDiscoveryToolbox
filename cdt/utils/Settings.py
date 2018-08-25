"""The ``cdt.utils.Settings`` module defines the settings used in the toolbox,
such as the default hardware parameters; and the tools to autodetect the 
hardware. All parameters are overridable by accessing the ``cdt.SETTINGS`` 
object, a unique instance of the ``cdt.utils.ConfigSettings`` class.

The various attributes of the ``cdt.SETTINGS`` configuration object are:

1. ``cdt.SETTINGS.NB_JOBS`` 
2. ``cdt.SETTINGS.GPU``
3. ``cdt.SETTINGS.default_device``
4. ``cdt.SETTINGS.autoset_config``
5. ``cdt.SETTINGS.verbose``
6. ``cdt.SETTINGS.r_is_available``

The hardware detection revolves around the presence of GPUs. If GPUs are 
present, ``cdt.SETTINGS.GPU`` is set to ``True`` and the number of jobs 
is set to the number of GPUs. Else the number of jobs is set to the number
of CPUs. Another test performed at startup is to check if an R framework 
is available, unlocking additional features of the toolbox.

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
    """Defining the class for the hardware/high level settings of the CDT.

    Attributes:
        NB_JOBS (int): Number of parallel jobs that can be executed on current
            hardware.
        GPU (int): The number of available GPUs ; defaults to `0`.
        default_device (str): Default device used for pytorch jobs.
        r_is_available (bool): Sets itself to ``True`` if a R framework is
            detected.
        verbose (bool): Sets the verbosity of the toolbox algorithms.

    """

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
        self.GPU = 0
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
        """Get the default parameters as defined in the Settings instance.

        This function proceeds to seamlessly retrieve the argument to pass 
        through, depending on either it was overidden or not: If no argument
        was overridden in a function of the toolbox, the default argument will
        be set to ``None``, and this function will retrieve the default 
        parameters as defined by the ``cdt.SETTINGS`` 's attributes.

        It has two modes of processing:

        1. **kwargs for retrieving a single argument: ``get_default(argument_name=value)``.
        2. *args through a list of tuples of the shape ``('argument_name', value)`` to retrieve multiple values at once.
        """
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
            warnings.warn("Detecting CUDA devices : {}".format(devices))

    except KeyError:
        set_var.GPU = check_cuda_devices()
        set_var.NB_JOBS = set_var.GPU
        warnings.warn("Detecting {} CUDA devices.".format(set_var.GPU))
        if not set_var.GPU:
            warnings.warn("No GPU automatically detected. Setting SETTINGS.GPU to 0, " +
                          "and SETTINGS.NB_JOBS to cpu_count.")
            set_var.GPU = 0
            set_var.NB_JOBS = multiprocessing.cpu_count()

    return set_var


def check_cuda_devices():
    """Output some information on CUDA-enabled devices on your computer, 
    including current memory usage. Modified to only get number of devices.

    It's a port of https://gist.github.com/f0k/0d6431e3faa60bffc788f8b4daa029b1
    from C to Python with ctypes, so it can run without compiling
    anything. Note that this is a direct translation with no attempt to
    make the code Pythonic. It's meant as a general demonstration on how
    to obtain CUDA device information from Python without resorting to
    nvidia-smi or a compiled Python extension.


    .. note:: Author: Jan Schlüter, https://gist.github.com/63a664160d016a491b2cbea15913d549.git
    """
    import ctypes

    # Some constants taken from cuda.h
    CUDA_SUCCESS = 0

    libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        # raise OSError("could not load any of: " + ' '.join(libnames))
        return 0

    nGpus = ctypes.c_int()
    error_str = ctypes.c_char_p()

    result = cuda.cuInit(0)
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        # print("cuInit failed with error code %d: %s" % (result, error_str.value.decode()))
        return 0
    result = cuda.cuDeviceGetCount(ctypes.byref(nGpus))
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        # print("cuDeviceGetCount failed with error code %d: %s" % (result, error_str.value.decode()))
        return 0
    # print("Found %d device(s)." % nGpus.value)
    return nGpus.value


SETTINGS = ConfigSettings()
