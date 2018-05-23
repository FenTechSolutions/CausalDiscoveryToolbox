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


def parse_dataframe(df):
    df = df.applymap(lambda cell: np.fromstring(cell, dtype=np.float, sep=" "))
    return df


def read_data(pairs_path, info_path):
    df_pairs = parse_dataframe(pd.read_csv(pairs_path, index_col="SampleID"))
    df_info = pd.read_csv(info_path, index_col="SampleID")
    features = pd.concat([df_pairs, df_info], axis=1)
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


def write_predictions(pred_path, test, predictions):
    writer = csv.writer(open(pred_path, "w"), lineterminator="\n")
    rows = [x for x in zip(test.index, predictions)]
    writer.writerow(("SampleID", "Target"))
    writer.writerows(rows)


def main():
    if len(sys.argv) < 4:
        print("USAGE: python predict.py CEdata_test_pairs.csv CEdata_test_publicinfo.csv CEdata_test_predictions.csv")
        return -1
    pairs_path = sys.argv[1]
    info_path = sys.argv[2]
    pred_path = sys.argv[3]

    test = read_data(pairs_path, info_path)

    print("Loading the classifier")
    model = load_model("model.pkl")
    print("model.weights", model.weights)

    print("Extracting features")
    test = model.extract(test)

    print("Making predictions")
    predictions = model.predict(test)
    print("Writing predictions to file")
    write_predictions(pred_path, test[0::2], predictions[0::2])


if __name__ == "__main__":
    main()
