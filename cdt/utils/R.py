"""Loading R packages by using subprocess.

Checking if the packages are available
Author: Diviyan Kalainathan
"""

import os
import warnings
import fileinput
import subprocess
from shutil import copy, rmtree


def message_warning(msg, *a, **kwargs):
    """Ignore everything except the message."""
    return str(msg) + '\n'


warnings.formatwarning = message_warning
init = True


class DefaultRPackages(object):
    """Define the packages to be tested for import."""

    __slots__ = ("init",
                 "pcalg",
                 "kpcalg",
                 "bnlearn",
                 "D2C",
                 "SID",
                 "CAM")

    def __init__(self):
        """Init the values of the packages."""
        self.init = True
        self.pcalg = None
        self.kpcalg = None
        self.bnlearn = None
        self.D2C = None
        self.SID = None
        self.CAM = None
        self.init = False

    def __repr__(self):
        """Representation."""
        return str(["{}: {}".format(i, getattr(self, i)) for i in self.__slots__])

    def __str__(self):
        """For print purposes."""
        return str(["{}: {}".format(i, getattr(self, i)) for i in self.__slots__])

    def __getattribute__(self, name):
        """Test if libraries are available on the fly."""
        out = object.__getattribute__(self, name)
        if out is None and not object.__getattribute__(self, 'init'):
            availability = self.check_R_package(name)
            setattr(self, name, availability)
            return availability
        return out

    def check_R_package(self, package):
        """Execute a subprocess to check the package's availability."""
        test_package = not bool(launch_R_script("{}/R_templates/test_import.R".format(os.path.dirname(os.path.realpath(__file__))),                                      {"{package}": package}, verbose=True))
        return test_package


def launch_R_script(template, arguments, output_function=None, verbose=True):
    """Launch an R script, starting from a template and replacing text in file before execution."""
    os.makedirs('/tmp/cdt_R_scripts/')
    try:
        scriptpath = '/tmp/cdt_R_scripts/instance_{}'.format(os.path.basename(template))
        copy(template, scriptpath)

        with fileinput.FileInput(scriptpath, inplace=True) as file:
            for line in file:
                mline = line
                for elt in arguments:
                    mline = mline.replace(elt, arguments[elt])
                print(mline, end='')

        if output_function is None:
            output = subprocess.call("Rscript --vanilla {}".format(scriptpath), shell=True,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            if verbose:
                process = subprocess.Popen("Rscript --vanilla {}".format(scriptpath), shell=True)
            else:
                process = subprocess.Popen("Rscript --vanilla {}".format(scriptpath), shell=True,
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            process.wait()
            output = output_function()

    # Cleaning up
    except Exception as e:
        rmtree('/tmp/cdt_R_scripts/')
        raise e
    except KeyboardInterrupt:
        rmtree('/tmp/cdt_R_scripts/')
        raise KeyboardInterrupt
    rmtree('/tmp/cdt_R_scripts/')
    return output


RPackages = DefaultRPackages()
init = False
