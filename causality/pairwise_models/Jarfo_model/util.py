"""
Random permutation of a symmetrized database

"""

# Author: Jose A. R. Fonollosa <jarfo@yahoo.com>
#
# License: Apache, Version 2.0

import numpy as np

def random_permutation(x, y, seed=14777777):
    np.random.seed(seed)
    global_random_permutation = np.array(range(len(x)/2))
    np.random.shuffle(global_random_permutation)
    index = np.array(range(len(x)))
    index[0::2] = 2*global_random_permutation
    index[1::2] = 2*global_random_permutation + 1
    x.index = index
    y.index = index
    return x.sort(), y.sort()
