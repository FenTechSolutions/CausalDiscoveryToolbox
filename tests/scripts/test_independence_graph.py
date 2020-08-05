"""Test markov blanket recovering methods."""

import os
import pandas as pd
import networkx as nx
from cdt import SETTINGS
from cdt.independence.graph import (Glasso, ARD, DecisionTreeRegression,
                                    LinearSVRL2, HSICLasso,
                                    RFECVLinearSVR)
from cdt.utils.graph import remove_indirect_links

SETTINGS.NJOBS = 1


def init():
    return pd.read_csv('{}/../datasets/Example_graph_numdata.csv'.format(os.path.dirname(os.path.realpath(__file__)))).iloc[:50, :5]


def test_statistical_methods():
    data = init()
    for method in [Glasso, ARD, DecisionTreeRegression,
                   LinearSVRL2, HSICLasso,
                   RFECVLinearSVR]:
        model = method()
        # print(method)
        assert isinstance(model.predict(data), nx.Graph)
    return 0


def test_indirect_link_removal():
    data = init()
    umg = Glasso().predict(data)
    for method in ["nd", "clr", "aracne"]:
        assert isinstance(remove_indirect_links(umg, alg=method), nx.Graph)
    return 0


if __name__ == '__main__':
    print("Statistical Methods")
    test_statistical_methods()
    print("test_indirect_link_removal")
    test_indirect_link_removal()
