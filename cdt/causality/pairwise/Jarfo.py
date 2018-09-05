"""
Jarfo causal inference model
Author : José AR Fonollosa
Ref : Fonollosa, José AR, "Conditional distribution variability measures for causality detection", 2016.
"""

from .Jarfo_model import train
from .model import PairwiseModel


class Jarfo(PairwiseModel):
    def __init__(self):
        super(Jarfo, self).__init__()

    def fit(self, df, tar):
        self.model = train.train(df, tar)

    def predict_proba(self, a, b):
        pass
