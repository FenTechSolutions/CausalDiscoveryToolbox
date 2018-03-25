"""Loading R packages by using subprocess.

Checking if the packages are available
Author: Diviyan Kalainathan
"""

import os
import warnings
import fileinput
from subprocess import call, DEVNULL
from shutil import copy, rmtree


def message_warning(msg, *a):
    """Ignore everything except the message."""
    return str(msg) + '\n'


warnings.formatwarning = message_warning


class DefaultRPackages(object):
    """Define the packages to be tested for import."""

    __slots__ = ("pcalg",
                 "kpcalg",
                 "bnlearn",
                 "D2C",
                 "SID")

    def __init__(self):
        """Init the values of the packages."""
        self.pcalg = False
        self.kpcalg = False
        self.bnlearn = False
        self.D2C = False
        self.SID = False


def check_R_packages(packages):
    """Execute a subprocess to check the packages' availability."""
    for i in packages.__slots__:
        setattr(packages, i,
                not bool(launch_R_script("{}/R_templates/test_import.R".format(os.path.dirname(os.path.realpath(__file__))),
                                         {"{package}": i}, verbose=False)))


def launch_R_script(template, arguments, verbose=True):
    """Launch an R script, starting from a template and replacing text in file before execution."""
    os.makedirs('/tmp/cdt_R_scripts/')
    try:
        scriptpath = '/tmp/cdt_R_scripts/instance_{}'.format(os.path.basename(template))
        copy(template, scriptpath)

        with fileinput.FileInput(scriptpath, inplace=True) as file:
            for line in file:
                for elt in arguments:
                    print(line.replace(elt, arguments[elt]), end='')
        if verbose:
            output = call("Rscript --vanilla {}".format(scriptpath),
                          shell=True)
        else:
            output = call("Rscript --vanilla {}".format(scriptpath),
                          shell=True, stdout=DEVNULL, stderr=DEVNULL)

    # Cleaning up
    except Exception as e:
        rmtree('/tmp/cdt_R_scripts/')
        raise e

    rmtree('/tmp/cdt_R_scripts/')
    return output


RPackages = DefaultRPackages()
check_R_packages(RPackages)
