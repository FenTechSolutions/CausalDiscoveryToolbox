import cdt
import pandas as pd
import sys

# Params
cdt.SETTINGS.GPU = True
cdt.SETTINGS.NB_GPU = 4
cdt.SETTINGS.NB_JOBS = 8

#Setting for CGNN-Fourier
cdt.SETTINGS.use_Fast_MMD = False
cdt.SETTINGS.NB_RUNS = 8
cdt.SETTINGS.NB_MAX_RUNS = 32

cdt.SETTINGS.nb_run_feature_selection = 16
cdt.SETTINGS.regul_param = 0.004
cdt.SETTINGS.threshold_UMG = 0.13



datafile = sys.argv[1]
umg_file = datafile + "_umg.csv"
pairwise_file = datafile + "_pairwise_predictions.csv"

print("Processing " + datafile + "...")
df_data = pd.read_csv(datafile, sep = '\t')
df_umg = pd.read_csv(umg_file, index_col = False)
df_pairwise = pd.read_csv(pairwise_file, index_col = False)

umg = cdt.UndirectedGraph(df_umg)
#print(umg)
p_directed_graph = cdt.DirectedGraph(df_pairwise, skeleton = umg)
#print(p_directed_graph)

#FSGNN = cdt.independence.graph.FSGNN()
#umg = FSGNN.create_skeleton_from_data(df_data)
#umg_res = pd.DataFrame(umg.list_edges(descending=True), columns=['Cause', 'Effect', 'Score'])
#umg_res.to_csv(datafile + "_umg.csv")


#GNN = cdt.causality.pairwise.GNN(backend="TensorFlow")
#p_directed_graph = GNN.orient_graph(df_data, umg, printout=datafile + '_printout.csv')
#gnn_res = pd.DataFrame(p_directed_graph.list_edges(descending=True), columns=['Cause', 'Effect', 'Score'])
#gnn_res.to_csv(datafile + "_pairwise_predictions.csv")




#CGNN = cdt.causality.graph.CGNN(backend="TensorFlow")
#directed_graph = CGNN.orient_directed_graph(df_data, p_directed_graph)
#cgnn_res = pd.DataFrame(directed_graph.list_edges(descending=True), columns=['Cause', 'Effect', 'Score'])
#cgnn_res.to_csv(datafile + "_predictions_CGNN.csv")


CGNN_confounders = cdt.causality.graph.CGNN_confounders(backend="TensorFlow")
directed_graph = CGNN_confounders.orient_directed_graph(df_data, p_directed_graph)
cgnn_res = pd.DataFrame(directed_graph.list_edges(descending=True), columns=['Cause', 'Effect', 'Score'])
cgnn_res.to_csv(datafile + "_predictions_CGNN_remove_edge.csv")



print('Processed ' + datafile)


