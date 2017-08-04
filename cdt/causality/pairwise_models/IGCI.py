# ToDo : Implement
from .model import Pairwise_Model


class IGCI(Pairwise_Model):
    def __init__(self):
        super(IGCI, self).__init__()

    def predict_proba(self, a, b):
        return 0