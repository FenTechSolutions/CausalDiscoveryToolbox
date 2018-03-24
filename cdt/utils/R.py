"""Loading R packages by using subprocess.

Checking if the packages are available
Author: Diviyan Kalainathan
"""

import warnings
import os
from subprocess import call, DEVNULL


def message_warning(msg, *a):
    """Ignore everything except the message."""
    return str(msg) + '\n'


warnings.formatwarning = message_warning


class DefaultRPackages(object):
    """Define the packages to be tested for import."""

    __slots__ = ("pcalg",
                 "kpcalg",
                 "bnlearn")

    def __init__(self):
        """Init the values of the packages."""
        self.pcalg = False
        self.kpcalg = False
        self.bnlearn = False


def check_R_packages(packages):
    """Execute a subprocess to check the packages' availability."""
    for i in packages.__slots__:
        setattr(packages, i,
                not bool(call("Rscript --vanilla {}/R_scripts/test_{}.R".format(os.path.dirname(os.path.realpath(__file__)), i),
                              shell=True, stdout=DEVNULL, stderr=DEVNULL)))


RPackages = DefaultRPackages()
check_R_packages(RPackages)
