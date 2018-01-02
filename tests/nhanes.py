import cdt
import time
import pandas as pd


#Hardware parameters
cdt.SETTINGS.GPU = False
cdt.SETTINGS.NB_JOBS = 1
#Settings for CGNN
cdt.CGNN_SETTINGS.use_Fast_MMD = False
cdt.CGNN_SETTINGS.NB_RUNS = 1
cdt.CGNN_SETTINGS.NB_MAX_RUNS = 1

#Settings for Feature Selection
cdt.CGNN_SETTINGS.nb_run_feature_selection = 1
cdt.CGNN_SETTINGS.regul_param = 0.006
cdt.SETTINGS.threshold_UMG = 0.16

cdt.CGNN_SETTINGS.nb_epoch_train_feature_selection = 1
cdt.CGNN_SETTINGS.nb_epoch_eval_weights = 1

cdt.CGNN_SETTINGS.train_epochs = 1
cdt.CGNN_SETTINGS.test_epochs = 1

cdt.CGNN_SETTINGS.max_nb_points = 10



data = pd.read_csv("data/filteredData.csv", sep = ";", index_col = 0)

# the data has 16 615 sample which takes too much time 
# i sampled 500 elements like the Lucas dataset

# data =data.sample(500)

# del data['Unnamed: 0']

# sex: 1 for male, 2 for female
# ethnicity(eth): 1,2,3,4,5 for respectively MexAm,OthHis,NHiWhi,NHiBl,Oth
print(data.head())



from cdt.independence.graph import FSGNN


type_variables = {}
type_variables["age"] = "Categorical"
type_variables["sex"] = "Categorical"
type_variables["ses"] = "Numerical"
type_variables["eth"] = "Categorical"
type_variables["psu"] = "Categorical"
type_variables["stra"] = "Categorical"
type_variables["bmi"] = "Numerical"
type_variables["wt"] = "Numerical"
type_variables["bmi_two_class"] = "Categorical"



Fsgnn = FSGNN()
start_time = time.time()
ugraph = Fsgnn.create_skeleton_from_data(data)
print("--- Execution time : %s seconds ---" % (time.time() - start_time))
ugraph.plot()
# List results
pd.DataFrame(ugraph.list_edges())


from cdt.causality.graph import CGNN




Cgnn = CGNN(backend="TensorFlow")
start_time = time.time()
dgraph = Cgnn.predict(data, graph=ugraph, type_variables = type_variables)
print("--- Execution time : %s seconds ---" % (time.time() - start_time))

# Plot the output graph
dgraph.plot()
# Print output results : 
print(dgraph.list_edges())