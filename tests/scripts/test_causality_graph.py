"""Test pairwise causal Discovery models."""

import os
import pandas as pd
import networkx as nx
from cdt import SETTINGS
from cdt.utils.io import read_causal_pairs
from cdt.causality.graph import (CAM, GS, GIES, IAMB,
                                 Fast_IAMB, PC, LiNGAM, SAM)

data_graph = pd.read_csv('{}/../datasets/Example_graph_numdata.csv'.format(os.path.dirname(os.path.realpath(__file__)))).iloc[:50, :5]

SETTINGS.verbose = False

def test_graph():
    for method in [CAM, GS, GIES,  IAMB, Fast_IAMB,
                   PC, LiNGAM]:
        print(method)
        m = method()
        assert isinstance(m.predict(data_graph), nx.DiGraph)
    return 0


def test_SAM():
    m = SAM(train_epochs=10, test_epochs=10, nh=10, dnh=10)
    assert isinstance(m.predict(data_graph, nruns=1, njobs=1), nx.DiGraph)
    return 0


if __name__ == "__main__":
    test_graph()
    test_SAM()
