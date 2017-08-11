# ToDo : Implement
from .model import Pairwise_Model


class PNL(Pairwise_Model):
    def __init__(self):
        super(PNL, self).__init__()

    def predict_proba(self, a, b):
        return 0