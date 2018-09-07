"""
Jarfo causal inference model
Author : José AR Fonollosa
Ref : Fonollosa, José AR, "Conditional distribution variability measures for causality detection", 2016.
"""

from pandas import DataFrame
from .Jarfo_model import train
from .model import PairwiseModel
from .Jarfo_model import predict

class Jarfo(PairwiseModel):
    def __init__(self):
        super(Jarfo, self).__init__()

    def fit(self, df, tar):
        self.model = train.train(df, tar)

    def predict_dataset(self, df):
        if self.model is None:
            raise AssertionError("Model has not been trained before predictions")
        return predict.predict(df, self.model)

    def predict_proba(self, a, b):
        return self.predict_dataset(DataFrame([["pair1", a, b]]))
