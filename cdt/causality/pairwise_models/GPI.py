# ToDo : Implement
from .model import Pairwise_Model


class GPI(Pairwise_Model):
    def __init__(self):
        super(GPI, self).__init__()

    def predict_proba(self, a, b):
        return 0
