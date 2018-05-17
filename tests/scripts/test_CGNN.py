"""Test script for CGNN."""

# Import libraries
import cdt
from cdt.independence.graph import FSGNN
from cdt.causality.graph import CGNN
from cdt import SETTINGS
import networkx as nx
import time
# A warning on R libraries might occur. It is for the use of the r libraries that could be imported into the framework
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

SETTINGS.GPU = True

print(SETTINGS.GPU)
print(SETTINGS.GPU_LIST)
print(SETTINGS.default_device)
data = pd.read_csv('{}/../datasets/NUM_LUCAS.csv'.format(os.path.dirname(os.path.realpath(__file__))))
solution = cdt.utils.read_list_edges('{}/../datasets/Lucas_graph.csv'.format(os.path.dirname(os.path.realpath(__file__))))


# Finding the structure of the graph
Fsgnn = FSGNN()

start_time = time.time()
ugraph = Fsgnn.predict(data, train_epochs=20, test_epochs=20, threshold=1e-2, l1=0.006)
print("--- Execution time : %4.4s seconds ---" % (time.time() - start_time))
# List results
print(pd.DataFrame(list(ugraph.edges(data='weight'))))
# print(nx.adj_matrix(ugraph).todense().shape)
# Orient the edges of the graph

Cgnn = CGNN()
start_time = time.time()
dgraph = Cgnn.predict(data, graph=ugraph, nb_runs=1, nb_max_runs=1, train_epochs=2, test_epochs=2)
print("--- Execution time : %4.4s seconds ---" % (time.time() - start_time))

# Plot the output graph
# Print output results :
print(pd.DataFrame(list(dgraph.edges(data='weight')), columns=['Cause', 'Effect', 'Score']))
