import cdt
import pandas as pd
import sys

# Params
cdt.SETTINGS.GPU = True
cdt.SETTINGS.NB_GPU = 4
cdt.SETTINGS.NB_JOBS = 4

#Setting for CGNN-Fourier
cdt.SETTINGS.use_Fast_MMD = False
cdt.SETTINGS.NB_RUNS = 4
cdt.SETTINGS.NB_MAX_RUNS = 8

cdt.SETTINGS.nb_run_feature_selection = 1
cdt.SETTINGS.regul_param = 0.004
cdt.SETTINGS.threshold_UMG = 0.15

cdt.SETTINGS.train_epochs = 100
cdt.SETTINGS.test_epochs = 100

#cdt.SETTINGS.max_nb_points = 1500


datafile = sys.argv[1]

print("Processing " + datafile + "...")
df_data = pd.read_csv(datafile)


FSGNN = cdt.independence.graph.FSGNN()

umg = FSGNN.create_skeleton_from_data(df_data)

umg_res = pd.DataFrame(umg.list_edges(descending=True), columns=['Cause', 'Effect', 'Score'])
umg_res.to_csv(datafile + "_umg.csv")


GNN = cdt.causality.pairwise.GNN(backend="TensorFlow")
p_directed_graph = GNN.orient_graph(df_data, umg, printout=datafile + '_printout.csv')
gnn_res = pd.DataFrame(p_directed_graph.list_edges(descending=True), columns=['Cause', 'Effect', 'Score'])
gnn_res.to_csv(datafile + "_pairwise_predictions.csv")

CGNN = cdt.causality.graph.CGNN(backend="TensorFlow")
directed_graph = CGNN.orient_directed_graph(df_data, p_directed_graph)
cgnn_res = pd.DataFrame(directed_graph.list_edges(descending=True), columns=['Cause', 'Effect', 'Score'])
cgnn_res.to_csv(datafile + "_predictions_CGNN.csv")

CGNN = cdt.causality.graph.CGNN(backend="TensorFlow")
directed_graph = CGNN.orient_directed_graph(df_data, p_directed_graph)
cgnn_res = pd.DataFrame(directed_graph.list_edges(descending=True), columns=['Cause', 'Effect', 'Score'])
cgnn_res.to_csv(datafile + "_predictions_CGNN.csv")


print('Processed ' + datafile)


