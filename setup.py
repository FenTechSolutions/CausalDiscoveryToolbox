# -*- coding: utf-8 -*-
# Copyright (C) 2016 Diviyan Kalainathan
# Licence: Apache 2.0

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages


def setup_package():
    setup(name='cdt',
          version='0.2.1',
          description='A Toolbox for causal graph inference',
          packages=find_packages(exclude=['examples', 'tests', 'tests.*']),
          url='https://github.com/Diviyan-Kalainathan/CausalDiscoveryToolbox',
          package_data={'': ['**/*.R']},
          include_package_data=True,
          author='Diviyan Kalainathan',
          author_email='diviyan.kalainathan@lri.fr',
          license='Apache 2.0')


if __name__ == '__main__':
    setup_package()
