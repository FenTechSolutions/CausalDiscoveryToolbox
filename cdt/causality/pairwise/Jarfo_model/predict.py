"""
Cause-effect direction prediction using the model saved in model.pkl

"""

# Author: Jose A. R. Fonollosa <jarfo@yahoo.com>
#
# License: Apache, Version 2.0

import sys
import csv
import numpy as np
import pandas as pd
import _pickle as pickle


def load_model(model_path, verbose=True):
    m = pickle.load(open(model_path))
    return m


def predict(df, model):
    df.columns = ["A", "B"]
    # print(df)
    df2 = model.extract(df)
    # print(df2)
    return model.predict(df2)
