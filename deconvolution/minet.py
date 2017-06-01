"""
Porting of the minet R package
Authors : ADD_AUTHORS
Paper : ADD_SOURCE

NOTE : Maybe wrapper on the R package ?
"""

import numpy as np

def clr(M):
    R = np.zeros((M.shape))
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
