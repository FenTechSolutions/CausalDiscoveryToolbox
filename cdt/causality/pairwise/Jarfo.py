"""
Jarfo causal inference model
Author : José AR Fonollosa
Ref : Fonollosa, José AR, "Conditional distribution variability measures for causality detection", 2016.
"""

from .Jarfo_model import *
from .model import PairwiseModel


class Jarfo(PairwiseModel):
    def __init__(self):
        super(Jarfo, self).__init__()
        raise NotImplementedError

    def fit(self, df, tar):
        pass

    def predict_proba(self, a, b):
        pass
