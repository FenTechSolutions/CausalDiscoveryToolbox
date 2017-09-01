import cdt
import sys
import pandas as pd

# Params
cdt.SETTINGS.GPU = True
cdt.SETTINGS.NB_GPU = 2
cdt.SETTINGS.NB_JOBS = 4

#Setting for CGNN-Fourier
cdt.SETTINGS.use_Fast_MMD = True
cdt.SETTINGS.NB_RUNS = 4
cdt.SETTINGS.NB_MAX_RUNS = 64

#Setting for CGNN-MMD
# cdt.SETTINGS.use_Fast_MMD = False
# cdt.SETTINGS.NB_RUNS = 32


datafile = "Example_graph_numdata.csv"
skeletonfile = "Example_graph_skeleton.csv"

print("Processing " + datafile + "...")
undirected_links = pd.read_csv(skeletonfile)

umg = cdt.UndirectedGraph(undirected_links)
data = pd.read_csv(datafile)

GNN = cdt.causality.pairwise.GNN(backend="TensorFlow")
p_directed_graph = GNN.orient_graph(data, umg, printout=datafile + '_printout.csv')
gnn_res = pd.DataFrame(p_directed_graph.list_edges(descending=True), columns=['Cause', 'Effect', 'Score'])
gnn_res.to_csv(datafile + "_pairwise_predictions.csv")

CGNN = cdt.causality.graph.CGNN(backend="TensorFlow")
directed_graph = CGNN.orient_directed_graph(data, p_directed_graph)
cgnn_res = pd.DataFrame(directed_graph.list_edges(descending=True), columns=['Cause', 'Effect', 'Score'])
cgnn_res.to_csv(datafile + "_predictions.csv")

print('Processed ' + datafile)
