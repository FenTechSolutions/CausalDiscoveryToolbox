import warnings
from .Settings import SETTINGS


class DefaultRPackages(object):
    __slots__ = ("pcalg",
                 "kpcalg",
                 "minet",
                 "bnlearn")

    def __init__(self):  # Define here the default values of the parameters
        self.pcalg = None
        self.minet = None
        self.kpcalg = None
        self.bnlearn = None


def load_r_wrapper():
    try:
        import rpy2
        try:
            import readline
            import rpy2.robjects
            from rpy2.robjects.packages import importr
            import rpy2.robjects.numpy2ri
            rpy2.robjects.numpy2ri.activate()
            RPackages.minet = importr('minet')
            RPackages.pcalg = importr("pcalg")
            RPackages.kpcalg = importr("kpcalg")
            RPackages.bnlearn = importr("bnlearn")
            SETTINGS.r_is_available = True
        except rpy2.rinterface.RRuntimeError as e:
            SETTINGS.r_is_available = False
            warnings.warn("R wrapper is not available : {}".format(e))
    except ImportError as e:
        SETTINGS.r_is_available = False
        warnings.warn("R wrapper is not available : {}".format(e))


def default_translation(rname):
    return rname.replace('.', '_')

RPackages = DefaultRPackages()
load_r_wrapper()
