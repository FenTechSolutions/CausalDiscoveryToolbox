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



    datafile = "graph/G2_v1_numdata.tab"

    datafile = sys.argv[1]

    cdt.CGNN_SETTINGS.asymmetry_param = 0.01
    cdt.CGNN_SETTINGS.h_layer_dim = 30

    type_FS = int(sys.argv[2])


    df_data = pd.read_csv(datafile, sep = '\t')
    df_data = pd.DataFrame(scale(df_data.as_matrix()), columns= df_data.columns)

    if(type_FS == 0):
        cdt.SETTINGS.threshold_UMG = 300
        RRelief = cdt.independence.graph.RRelief()
        umg = RRelief.predict(df_data)
        umg_res = pd.DataFrame(umg.list_edges(descending=True), columns=['Cause', 'Effect', 'Score'])
        umg_res.to_csv(datafile + "_umg_RRelief.csv")
    elif(type_FS == 1):
        cdt.SETTINGS.threshold_UMG = 300
        HSICLasso = cdt.independence.graph.RRelief()
        umg = HSICLasso.predict(df_data)
        umg_res = pd.DataFrame(umg.list_edges(descending=True), columns=['Cause', 'Effect', 'Score'])
        umg_res.to_csv(datafile + "_umg_HSICLasso.csv")
    elif(type_FS == 2):
        cdt.SETTINGS.threshold_UMG = 0.13
        FSGNN = cdt.independence.graph.FSGNN()
        umg = FSGNN.predict(df_data)
        umg_res = pd.DataFrame(umg.list_edges(descending=True), columns=['Cause', 'Effect', 'Score'])
        umg_res.to_csv(datafile + "_umg_FSGNN.csv")

    print(umg)

    type_variables = {}
    for node in df_data.columns:
        type_variables[node] = "Numerical"


    CGNN_decomposable = cdt.causality.graph.CGNN_decomposable(backend="TensorFlow")
    directed_graph = CGNN_decomposable.orient_undirected_graph(df_data, type_variables, umg, 2)

    cgnn_res = pd.DataFrame(directed_graph.list_edges(descending=True), columns=['Cause', 'Effect', 'Score'])
    cgnn_res.to_csv(datafile + "_predictions" + str(cdt.CGNN_SETTINGS.asymmetry_param) + "_" + str(cdt.CGNN_SETTINGS.h_layer_dim) + ".csv")

    print('Processed ' + datafile)
