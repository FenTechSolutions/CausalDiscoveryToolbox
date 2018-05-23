"""Test markov blanket recovering methods."""

import os
import pandas as pd
import networkx as nx
from cdt import SETTINGS
from cdt.independence.graph import (Glasso, ARD_Regression, DecisionTree_regressor,
                                    LinearSVR_L2, RandomizedLasso_model,
                                    RFECV_linearSVR, RRelief)
from cdt.independence.graph.remove_undirect_links import remove_indirect_links

SETTINGS.NB_JOBS = 1


def init():
    return pd.read_csv('{}/../datasets/Example_graph_numdata.csv'.format(os.path.dirname(os.path.realpath(__file__)))).iloc[:50, :5]


def test_statistical_methods():
    data = init()
    for method in [Glasso, ARD_Regression, DecisionTree_regressor,
                   LinearSVR_L2, RandomizedLasso_model,
                   RFECV_linearSVR, RRelief]:
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
