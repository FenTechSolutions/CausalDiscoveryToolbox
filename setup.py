# -*- coding: utf-8 -*-
# Copyright (C) 2016 Diviyan Kalainathan
# Licence: Apache 2.0

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages


with open('README.md', encoding="utf-8") as f:
    long_description = f.read()


def setup_package():
    setup(name='cdt',
          version='0.5.18',
          description='A Toolbox for causal graph inference',
          long_description="Documentation available at https://fentechsolutions.github.io/CausalDiscoveryToolbox/html/index.html",
          long_description_content_type="text/markdown",
          packages=find_packages(exclude=['examples', 'tests', 'tests.*']),
          url='https://github.com/FenTechSolutions/CausalDiscoveryToolbox',
          package_data={'': ['**/*.R', '**/*.csv']},
          install_requires=['numpy', 'scipy', 'scikit-learn',
                            'joblib', 'pandas', 'statsmodels',
                            'networkx', 'skrebate', 'tqdm', 'GPUtil', 'requests'],
          include_package_data=True,
          author='Diviyan Kalainathan',
          author_email='diviyan.kalainathan@lri.fr',
          license='Apache 2.0')


if __name__ == '__main__':
    setup_package()
