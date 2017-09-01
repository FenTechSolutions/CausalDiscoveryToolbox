"""
Portage of the R 'minet' package
"""

import numpy as np
import operator
from .model import DeconvolutionModel


class CLR(DeconvolutionModel):
    def __init__(self):
        super(CLR, self).__init__()

    def create_skeleton_from_data(self, data, **kwargs):
        raise ValueError("CLR does not create a skeleton from raw data but from an correlation/MI matrix.")

    def predict(self, M, **kwargs):
        R = np.zeros(M.shape)
        I = [[0, 0] for i in range(M.shape[0])]
        for i in range(M.shape[0]):
            mu_i = np.mean(M[i, :])
            sigma_i = np.std(M[i, :])
            I[i] = [mu_i, sigma_i]

        for i in range(M.shape[0]):
            for j in range(i + 1, M.shape[0]):
                z_i = np.max([0, (M[i, j] - I[i][0]) / I[i][0]])
                z_j = np.max([0, (M[i, j] - I[j][0]) / I[j][0]])
                R[i, j] = np.sqrt(z_i**2 + z_j**2)
                R[j, i] = R[i, j]  # Symmetric

        return R


class MRNet(DeconvolutionModel):
    def __init__(self):
        super(MRNet, self).__init__()

    def create_skeleton_from_data(self, data, **kwargs):
        raise ValueError("MRNet does not create a skeleton from raw data but from an correlation/MI matrix.")

    def predict(self, m, **kwargs):
        pass


class Aracne(DeconvolutionModel):
    def __init__(self):
        super(Aracne, self).__init__()

    def create_skeleton_from_data(self, data, **kwargs):
        raise ValueError("Aracne does not create a skeleton from raw data but from an correlation/MI matrix.")

    def predict(self, m, **kwargs):
        I0 = kwargs.get('I0', 0.0)  # No default thresholding
        W0 = kwargs.get('W0', 0.05)

        # thresholding
        m = np.where(m > I0, m, 0)

        # Finding triplets and filtering them
        for i in range(m.shape[0]-2):
            for j in range(i+1, m.shape[0]-1):
                for k in range(j+1, m.shape[0]):
                    triplet = [m[i, j], m[j, k], m[i, k]]
                    min_index, min_value = min(enumerate(triplet), key=operator.itemgetter(1))
                    if 0 < min_value < W0:
                        if min_index == 0:
                            m[i, j] = m[j, i] = 0.
                        elif min_index == 1:
                            m[j, k] = m[k, j] = 0.
                        else:
                            m[i, k] = m[k, i] = 0.
        return m




