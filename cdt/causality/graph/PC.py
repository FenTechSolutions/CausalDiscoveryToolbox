from ...utils.Settings import SETTINGS
from ...utils.R import RPackages
from .model import GraphModel
from ...utils.Graph import UndirectedGraph
from pandas import DataFrame
import warnings


class PC(GraphModel):
    def __init__(self):
        super(PC, self).__init__()

    def orient_undirected_graph(self, data, umg, **kwargs):
        if not SETTINGS.r_is_available:
            raise RuntimeError("R framework is not available")
        return 0

    def orient_directed_graph(self, data, dag, **kwargs):
        m, nodes = dag.adjacency_matrix()
        skeleton = UndirectedGraph(DataFrame(m, columns=nodes))
        warnings.warn("PC algorithm is run on the skeleton of the DAG")
        return self.orient_undirected_graph(data, skeleton, **kwargs)

    def create_graph_from_data(self, data, **kwargs):
        if not SETTINGS.r_is_available:
            raise RuntimeError("R framework is not available")

        return 0