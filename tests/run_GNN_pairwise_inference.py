import cdt
from cdt.utils import CCEPC_PairsFileReader as CC
import pandas as pd

# Params
cdt.SETTINGS.GPU = False
cdt.SETTINGS.NB_GPU = 1
cdt.SETTINGS.NB_JOBS = 1
cdt.SETTINGS.NB_MAX_RUNS = 32
#Setting for CGNN-Fourier
cdt.SETTINGS.use_Fast_MMD = True
cdt.SETTINGS.NB_RUNS = 1

#Setting for CDT-MMD
cdt.SETTINGS.use_Fast_MMD = True
cdt.SETTINGS.NB_RUNS = 1


cdt.SETTINGS.max_nb_points = 500


datafile = "Example_pairwise_pairs.csv"

print("Processing " + datafile + "...")
data = CC(datafile, scale=True)
model = cdt.causality.pairwise.GNN(backend="TensorFlow")
predictions = model.predict_dataset(data, printout=datafile + '_printout.csv')
predictions = pd.DataFrame(predictions, columns=["Predictions", "P-value"])

print('Processed ' + datafile)
predictions.to_csv(datafile + "_predictions_GNN.csv")
