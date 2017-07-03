"""
Cause-effect model training

"""

# Author: Jose A. R. Fonollosa <jarfo@yahoo.com>
#
# License: Apache, Version 2.0

import sys
import data_io
import numpy as np
import estimator as ce
import features as f
import pandas as pd
from scipy.optimize import fmin
import _pickle as pickle
import util


MODEL = ce.CauseEffectSystemCombination
MODEL_PARAMS = {'weights': [0.383, 0.370, 0.247], 'n_jobs': -1}


def train(df, tar, save = False):
    set1 = 'train' if len(sys.argv) < 2 else sys.argv[1]
    # set2 = [] if len(sys.argv) < 3 else sys.argv[2:]
    train_filter = None

    model = MODEL(**MODEL_PARAMS)

    print("Reading in training data " + set1)
    train = df
    print("Extracting features")
    train = model.extract(train)
    if save:
        print("Saving train features")
        data_io.write_data(set1, train)
    # target = data_io.read_target(set1)


    # Data selection
    train, target = util.random_permutation(train, tar)
    train_filter = None

    if train_filter is not None:
        train = train[train_filter]
        target = target[train_filter]

    print("Training model with optimal weights")
    X = pd.concat([train])
    y = np.concatenate((tar))
    model.fit(X, y)
    if save:
        model_path = "model.pkl"
        print("Saving model", model_path)
        data_io.save_model(model, model_path)
    return model

