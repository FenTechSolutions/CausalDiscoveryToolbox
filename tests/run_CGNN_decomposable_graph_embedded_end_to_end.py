import cdt
import pandas as pd
import sklearn.datasets as dataset
import sys
from sklearn.preprocessing import scale


if __name__ == '__main__':

    # Params
    cdt.SETTINGS.GPU = True
    cdt.SETTINGS.NB_JOBS = 2

    cdt.SETTINGS.GPU_LIST = ["0","1"]

    #Setting for CGNN-Fourier
    cdt.CGNN_SETTINGS.use_Fast_MMD = False
    cdt.CGNN_SETTINGS.NB_RUNS = 10

    cdt.CGNN_SETTINGS.h_layer_dim = 30


    datafile = sys.argv[1]

    cdt.CGNN_SETTINGS.asymmetry_param = sys.argv[2]
    cdt.CGNN_SETTINGS.regul_param = sys.argv[3]


    df_data = pd.read_csv(datafile, sep = '\t')
    df_data = pd.DataFrame(scale(df_data.as_matrix()), columns= df_data.columns)

    type_variables = {}
    for node in df_data.columns:
        type_variables[node] = "Numerical"

    CGNN_decomposable = cdt.causality.graph.CGNN_decomposable(backend="TensorFlow")
    directed_graph = CGNN_decomposable.create_graph_from_data(df_data)

    cgnn_res = pd.DataFrame(directed_graph.list_edges(descending=True), columns=['Cause', 'Effect', 'Score'])
    cgnn_res.to_csv(datafile + "_predictions" + str(cdt.CGNN_SETTINGS.asymmetry_param) + "_" + str(cdt.CGNN_SETTINGS.h_layer_dim) + ".csv")

    print('Processed ' + datafile)
