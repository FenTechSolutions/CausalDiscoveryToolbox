import cdt
import pandas as pd
import sys

# Params
cdt.SETTINGS.GPU = True

cdt.SETTINGS.NB_JOBS = 2
#cdt.SETTINGS.GPU_LIST = ["0"]

#Setting for CGNN-Fourier
cdt.CGNN_SETTINGS.use_Fast_MMD = False
cdt.CGNN_SETTINGS.NB_RUNS = 2


# cdt.CGNN_SETTINGS.train_epochs = 2000
# cdt.CGNN_SETTINGS.test_epochs = 500


# datafile = sys.argv[1]
# skeletonfile = sys.argv[2]

datafile = "graph/G2_v1_numdata.tab"
skeletonfile = "graph/G2_v1_target.tab"

print("Processing " + datafile + "...")
undirected_links = pd.read_csv(skeletonfile, sep = '\t')
#undirected_links = pd.read_csv(skeletonfile)

umg = cdt.UndirectedGraph(undirected_links)
df_data = pd.read_csv(datafile, sep = '\t')
#df_data = pd.read_csv(datafile)

#df_type = pd.read_csv(type_file)
#df_block = pd.read_csv(save_block, sep = '\t')



#type_variables = {}
#for i in range(df_type.shape[0]):
#    type_variables[df_type["Node"].loc[i]] = df_type["Type"].loc[i]

type_variables = {}
for node in df_data.columns:
    type_variables[node] = "Numerical"


CGNN_decomposable = cdt.causality.graph.CGNN_decomposable(backend="TensorFlow")
directed_graph = CGNN_decomposable.orient_undirected_graph(df_data, type_variables, umg, 2)

cgnn_res = pd.DataFrame(directed_graph.list_edges(descending=True), columns=['Cause', 'Effect', 'Score'])
cgnn_res.to_csv(datafile + "_predictions.csv")

print('Processed ' + datafile)
