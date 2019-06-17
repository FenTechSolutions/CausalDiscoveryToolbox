""" Pair Generator.
Generates pairs X,Y of variables with their labels.
Author: Diviyan Kalainathan

.. MIT License
..
.. Copyright (c) 2018 Diviyan Kalainathan
..
.. Permission is hereby granted, free of charge, to any person obtaining a copy
.. of this software and associated documentation files (the "Software"), to deal
.. in the Software without restriction, including without limitation the rights
.. to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
.. copies of the Software, and to permit persons to whom the Software is
.. furnished to do so, subject to the following conditions:
..
.. The above copyright notice and this permission notice shall be included in all
.. copies or substantial portions of the Software.
..
.. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
.. IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
.. FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
.. AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
.. LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
.. OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
.. SOFTWARE.
"""
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd
from .causal_mechanisms import (LinearMechanism,
                                Polynomial_Mechanism,
                                SigmoidAM_Mechanism,
                                SigmoidMix_Mechanism,
                                GaussianProcessAdd_Mechanism,
                                GaussianProcessMix_Mechanism,
                                NN_Mechanism,
                                gmm_cause,
                                normal_noise)
from joblib import delayed, Parallel
from ..utils.Settings import SETTINGS


class CausalPairGenerator(object):
    """Generates Bivariate Causal Distributions.

    Args:
        causal_mechanism (str): currently implemented mechanisms:
            ['linear', 'polynomial', 'sigmoid_add',
            'sigmoid_mix', 'gp_add', 'gp_mix', 'nn'].
        noise (str or function): type of noise to use in the generative process
            ('normal', 'uniform' or a custom noise function).
        noise_coeff (float): Proportion of noise in the mechanisms.
        initial_variable_generator (function): Function used to init variables
            of the graph, defaults to a Gaussian Mixture model.

    Example:
        >>> from cdt.data import CausalPairGenerator
        >>> generator = ('linear')
        >>> data, labels = generator.generate(100, npoints=500)
        >>> generator.to_csv('generated_pairs')
    """

    def __init__(self, causal_mechanism, noise=normal_noise,
                 noise_coeff=.4, initial_variable_generator=gmm_cause):
        super(CausalPairGenerator, self).__init__()
        self.mechanism = {'linear': LinearMechanism,
                          'polynomial': Polynomial_Mechanism,
                          'sigmoid_add': SigmoidAM_Mechanism,
                          'sigmoid_mix': SigmoidMix_Mechanism,
                          'gp_add': GaussianProcessAdd_Mechanism,
                          'gp_mix': GaussianProcessMix_Mechanism,
                          'nn': NN_Mechanism}[causal_mechanism]

        self.noise = noise
        self.noise_coeff = noise_coeff
        self.initial_generator = initial_variable_generator

    def generate(self, npairs, npoints=500, rescale=True, njobs=None):
        """Generate Causal pairs, such that one variable causes the other.

        Args:
            npairs (int): Number of pairs of variables to generate.
            npoints (int): Number of data points to generate.
            rescale (bool): Rescale the output with zero mean and unit variance.
            njobs (int): Number of parallel jobs to execute. Defaults to
                cdt.SETTINGS.NJOBS

        Returns:
            tuple: (pandas.DataFrame, pandas.DataFrame) data and corresponding
            labels. The data is at the ``SampleID, a (numpy.ndarray) , b (numpy.ndarray))``
            format.
        """
        def generate_pair(npoints, label, rescale):
            root = self.initial_generator(npoints)[:, np.newaxis]
            cause = self.mechanism(1, npoints, self.noise,
                                   noise_coeff=self.noise_coeff)(root)
            effect = self.mechanism(1, npoints, self.noise,
                                    noise_coeff=self.noise_coeff)(cause).squeeze(1)
            cause = cause.squeeze(1)
            if rescale:
                cause = scale(cause)
                effect = scale(effect)
            return (cause, effect) if label == 1 else (effect, cause)

        njobs = SETTINGS.get_default(njobs=njobs)
        self.labels = (np.random.randint(2, size=npairs) - .5) * 2
        output = [generate_pair(npoints, self.labels[i], rescale) for i in range(npairs)]
        self.data = pd.DataFrame(output, columns=['A', 'B'])
        self.labels = pd.DataFrame(self.labels, dtype='int32', columns=['label'])
        return self.data, self.labels

    def to_csv(self, fname_radical, **kwargs):
        """
        Save data to the csv format by default, in two separate files.

        Optional keyword arguments can be passed to pandas.
        """
        if self.data is not None:
            self.data.to_csv(fname_radical+'_data.csv', index=False, **kwargs)
            self.labels.to_csv(fname_radical + '_labels.csv',
                                                       index=False, **kwargs)

        else:
            raise ValueError("Data has not yet been generated. \
                              Use self.generate() to do so.")
