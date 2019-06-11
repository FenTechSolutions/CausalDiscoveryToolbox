import os
import numpy as np
import pandas as pd
import networkx as nx
from cdt.causality.pairwise.Jarfo import Jarfo
from cdt.independence.graph import Glasso
from cdt.utils.io import read_causal_pairs
from cdt import SETTINGS
from cdt.data import load_dataset


SETTINGS.NJOBS = 1


train_data = read_causal_pairs("{}/../datasets/Example_pairwise_pairs.csv".format(os.path.dirname(os.path.realpath(__file__)))).iloc[:, :50]

train_target = pd.read_csv("{}/../datasets/Example_pairwise_targets.csv".format(os.path.dirname(os.path.realpath(__file__)))).iloc[:, :50].set_index("SampleID")

data_pairwise = read_causal_pairs("{}/../datasets/Example_pairwise_pairs.csv".format(os.path.dirname(os.path.realpath(__file__)))).iloc[:5, :50]

data_graph = pd.read_csv('{}/../datasets/Example_graph_numdata.csv'.format(os.path.dirname(os.path.realpath(__file__)))).iloc[:100, :5]

train_data = pd.concat([train_data]*5, ignore_index=True)
train_target = pd.concat([train_target]*5, ignore_index=True)
train_target.iloc[10:,:]=0
# print(train_target)
graph_skeleton = Glasso().predict(data_graph)


def test_pairwise():
    for method in [Jarfo]:  # Jarfo
        # print(method)
        m = method()
        if hasattr(m, "fit"):
            # print(train_data)
            m.fit(train_data, train_target)
        r = m.predict(data_pairwise)
        assert r is not None
        print(r)
    return 0


def test_graph():
    for method in [Jarfo]:  # Jarfo
        print(method)
        m = method()
        if hasattr(m, "fit"):
            m.fit(train_data, train_target)
        assert type(m.predict(data_graph, graph_skeleton)) == nx.DiGraph
    return 0


def test_tuebingen():
    data, labels = load_dataset('tuebingen')
    data = data[:30]
    labels = labels[:30]
    # print(labels)
    m = Jarfo()
    m.fit(data, labels[['Target']])
    r = m.predict(data)
    print(r)
    return 0


def test_categorical():
    data, labels = load_dataset('tuebingen')
    data = data[:10]
    for idx in range(10):
        data.iloc[idx, 0] = np.digitize(data.iloc[idx, 0],
                                        np.histogram(data.iloc[idx, 0])[1])
        data.iloc[idx, 1] = np.digitize(data.iloc[idx, 1],
                                        np.histogram(data.iloc[idx, 1])[1])
    labels = labels[:10]
    m = Jarfo()
    m.fit(data, labels[['Target']])
    r = m.predict(data)
    print(r)
    return 0



if __name__ == "__main__":
    # test_pairwise()
    # test_graph()
    test_tuebingen()
    test_categorical()
