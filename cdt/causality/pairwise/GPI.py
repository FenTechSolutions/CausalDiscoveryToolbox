# ToDo : Implement
from .model import PairwiseModel


class GPI(PairwiseModel):
    def __init__(self):
        super(GPI, self).__init__()

    def predict_proba(self, a, b):
        return 0
