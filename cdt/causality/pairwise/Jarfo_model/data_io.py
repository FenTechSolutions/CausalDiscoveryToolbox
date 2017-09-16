"""
File I/O Utilities.

"""

import csv
import json
import numpy as np
import os
import pandas as pd
import _pickle as pickle
import os.path
import time


class InfoArray(np.ndarray):
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0, strides=None, order=None, stype=None):
        obj = np.ndarray.__new__(subtype, shape, dtype, buffer, offset, strides, order)
        obj.stype = stype
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.stype = getattr(obj, 'stype', None)


def get_path(key):
    paths = json.loads(open("SETTINGS.json").read())
    return os.path.expandvars(paths[key])


def parse_dataframe(df):
    parse_cell = lambda cell: np.fromstring(cell, dtype=np.float, sep=" ")
    df = df.applymap(parse_cell)
    return df


def read_pairs(set):
    train_path = get_path(set + "_pairs_path")
    return parse_dataframe(pd.read_csv(train_path, index_col="SampleID"))


def read_target(set):
    path = get_path(set + "_target_path")
    df = pd.read_csv(path, index_col="SampleID")
    df = df.rename(columns=dict(zip(df.columns, ["Target", "Details"])))
    # Duplicate training sequences exchanging 'A' with 'B'
    df_inverse = df.copy()
    df_inverse.Target = -df.Target
    df_inverse.Details[df.Details == 1] = 2
    df_inverse.Details[df.Details == 2] = 1
    original_index = np.array(zip(df.index, df.index)).flatten()
    df = pd.concat([df, df_inverse])
    df.index = range(0, len(df), 2) + range(1, len(df), 2)
    df.sort(inplace=True)
    df.index = original_index
    df.index.name = "SampleID"
    return df


def read_info(set):
    path = get_path(set + "_info_path")
    return pd.read_csv(path, index_col="SampleID")


def read_data(set):
    try:
        path = get_path(set + "_features_path")
        features = pd.DataFrame();
        features = features.load(path);
    except (IOError, EOFError):
        df_pairs = read_pairs(set)
        df_info = read_info(set)
        features = pd.concat([df_pairs, df_info], axis=1)
        # Duplicate training sequences exchanging 'A' with 'B'
        features_inverse = features.copy()
        features_inverse['A'] = features['B']
        features_inverse['A type'] = features['B type']
        features_inverse['B'] = features['A']
        features_inverse['B type'] = features['A type']
        original_index = np.array(zip(features.index, features.index)).flatten()
        features = pd.concat([features, features_inverse])
        features.index = range(0, len(features), 2) + range(1, len(features), 2)
        features.sort(inplace=True)
        features.index = original_index
        features.index.name = "SampleID"
    return features


def write_data(set, features):
    path = get_path(set + "_features_path")
    features.save(path)


def write_predictions(out_path, train, predictions):
    writer = csv.writer(open(out_path, "w"), lineterminator="\n")
    rows = [x for x in zip(train.index, predictions)]
    writer.writerow(("SampleID", "Target"))
    writer.writerows(rows)


def read_predictions(in_path):
    return pd.read_csv(in_path, index_col="SampleID")


def read_solution(solution_path=None):
    if not solution_path:
        solution_path = get_path("solution_path")
    return pd.read_csv(solution_path, index_col="SampleID")


def save_model(model, out_path=None):
    if not out_path:
        out_path = get_path("model_path") if not out_path else out_path
    pickle.dump(model, open(out_path, "w"))


def load_model(in_path=None, verbose=True):
    if not in_path:
        in_path = get_path("model_path")
    m = pickle.load(open(in_path))
    if (verbose):
        print("Model filename:", in_path)
        print("Model date: %s" % time.ctime(os.path.getmtime(in_path)))
        print("Model type: %s" % type(m))
    return m


def read_submission():
    submission_path = get_path("submission_path")
    return read_predictions(submission_path)


def write_submission(valid, predictions, info=None):
    submission_path = get_path("submission_path")
    submission_path = submission_path if info is None else submission_path + "_" + info
    write_predictions(submission_path, valid, predictions)
