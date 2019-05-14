"""
Cause-effect model training

"""

# Author: Jose A. R. Fonollosa <jarfo@yahoo.com>
#
# License: Apache, Version 2.0

import sys
import numpy as np
from .estimator import CauseEffectSystemCombination
# import features as f
import pandas as pd
# from scipy.optimize import fmin
# import _pickle as pickle
from .util import random_permutation


MODEL = CauseEffectSystemCombination
MODEL_PARAMS = {'weights': [0.383, 0.370, 0.247], 'njobs': 1}


def train(df, tar):
    set1 = 'train' if len(sys.argv) < 2 else sys.argv[1]
    # set2 = [] if len(sys.argv) < 3 else sys.argv[2:]
    train_filter = None
    # if len(df) % 2:
    #    df.drop(df.tail(1).index, inplace=True)
    #    tar.drop(tar.tail(1).index, inplace=True)
    model = MODEL(**MODEL_PARAMS)
    print("Reading in training data " + set1)
    train = df
    print("Extracting features")
    train = model.extract(train)
    # if save:
    #     print("Saving train features")
    #     write_data(set1, train)
    # target = data_io.read_target(set1)


    # Data selection
    train, target = random_permutation(train, tar)
    train_filter = None
    if train_filter is not None:
        train = train[train_filter]
        target = target[train_filter]

    print("Training model with optimal weights")
    # print(train)
    # print(tar.values)
    X = pd.concat([train])
    y = np.concatenate((tar.values))
    model.fit(X, y)
    # if save:
    #     model_path = "model.pkl"
    #     print("Saving model", model_path)
    #     save_model(model, model_path)
    return model
