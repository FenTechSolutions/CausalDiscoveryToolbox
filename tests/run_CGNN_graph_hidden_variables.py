import cdt
import pandas as pd
from sklearn.preprocessing import scale

# Params
cdt.SETTINGS.GPU = True
cdt.SETTINGS.NB_GPU = 1
cdt.SETTINGS.NB_JOBS = 1

#Setting for CGNN-Fourier
cdt.SETTINGS.use_Fast_MMD = True
cdt.SETTINGS.NB_RUNS = 64 

#Setting for CGNN-MMD
# cdt.SETTINGS.use_Fast_MMD = False
#cdt.SETTINGS.NB_RUNS = 32

datafile = "Example_graph_confounders_numdata.csv"
skeletonfile = "Example_graph_confounders_skeleton.csv"


data = pd.read_csv(datafile)
skeleton_links = pd.read_csv(skeletonfile)

skeleton = cdt.UndirectedGraph(skeleton_links)

data = pd.DataFrame(scale(data),columns=data.columns)

GNN = cdt.causality.pairwise.GNN(backend="TensorFlow")
p_directed_graph = GNN.orient_graph_confounders(data, skeleton, printout= datafile +  '_printout.csv')

gnn_res = pd.DataFrame(p_directed_graph.list_edges(descending=True), columns=['Cause', 'Effect', 'Score'])
gnn_res.to_csv(datafile + "_pairwise_predictions.csv")
CGNN_confounders = cdt.causality.graph.CGNN_confounders(backend="TensorFlow")
directed_graph = CGNN_confounders.orient_directed_graph(data, p_directed_graph)
cgnn_res = pd.DataFrame(directed_graph.list_edges(descending=True), columns=['Cause', 'Effect', 'Score'])

cgnn_res.to_csv(datafile + "_confounders_predictions.csv")


