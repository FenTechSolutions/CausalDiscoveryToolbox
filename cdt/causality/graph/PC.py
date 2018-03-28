"""PC algorithm by C.Glymour & P.Sprites (REF, 2000).

Imported from the Pcalg package.
Author = Diviyan Kalainathan
"""
from ...utils.Settings import SETTINGS
from ...utils.R import RPackages, default_translation
from .model import GraphModel
from ...utils.Graph import UndirectedGraph, DirectedGraph
from pandas import DataFrame
import numpy as np
import warnings
if SETTINGS.r_is_available:
    import rpy2
    rpy2.robjects.r['options'](warn=-1)
    rpy2.robjects.numpy2ri.activate()
    from rpy2.robjects import r
    import rpy2.rlike.container as rlc


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
        """ Run the PC algorithm
        from the pcalg manual:
        :param data: DataFrame containing the data
        :param kwargs:indepMethod: Method used to test conditional independence.
        :param kwargs:indepTest: Predefined function for testing conditional independence. The function is internally
        called as indepTest(x,y,S,suffStat) and tests conditional independence of x and y given S. Here, x and y are
         variables, and S is a (possibly empty) vector of variables (all variables are denoted by their column numbers
        in the adjacency matrix). suffStat is a list containing all relevant elements for the conditional independence
         decisions. The return value of indepTest is the p-value of the test for conditional independence.
        :param kwargs:alpha: significance level (number in (0, 1) for the individual conditional independence tests.
        :param kwargs:labels: (optional) character vector of variable (or “node”) names. Typically preferred to
        specifying p.
        :param kwargs:p: (optional) number of variables (or nodes). May be specified if labels are not,
        in which case labels is set to 1:p.
        :param kwargs:numCores: number of processor cores to use for parallel computation. Only available for
        method = "stable.fast".
        :param kwargs:verbose: if TRUE, detailed output is provided.
        :param kwargs:fixedGaps: A logical matrix of dimension p*p. If entry [i,j] or [j,i] (or both) are TRUE,
        the edge i-j is removed before starting the algorithm. Therefore, this edge is
        guaranteed to be absent in the resulting graph.
        :param kwargs:fixedEdges: A logical matrix of dimension p*p. If entry [i,j] or [j,i] (or both) are TRUE,
        the edge i-j is never considered for removal. Therefore, this edge is guaranteed
        to be present in the resulting graph.
        :param kwargs:NAdelete: If indepTest returns NA and this option is TRUE, the corresponding edge is deleted.
        If this option is FALSE, the edge is not deleted.
        :param kwargs:m.max: Maximal size of the conditioning sets that are considered in the conditional independence
        tests.
        :param kwargs:u2pd: String specifying the method for dealing with conflicting information when trying
        to orient edges (see details below).
        :param kwargs:skel.method: Character string specifying method; the default, "stable" provides an orderindependent
        skeleton, see skeleton.
        conservative Logical indicating if the conservative PC is used. In this case, only option
        :param kwargs:u2pd: = "relaxed" is supported. Note that therefore the resulting object might
        not be extendable to a DAG.
        :param kwargs:maj.rule: Logical indicating that the triples shall be checked for ambiguity using a majority
        rule idea, which is less strict than the conservative PC algorithm. For more
        information, see details.
        :param kwargs:solve.confl: If TRUE, the orientation of the v-structures and the orientation rules work with
        lists for candidate sets and allow bi-directed edges to resolve conflicting edge
        orientations. In this case, only option u2pd = relaxed is supported. Note, that
        therefore the resulting object might not be a CPDAG because bi-directed edges
        might be present.
        """
        if not SETTINGS.r_is_available:
            raise RuntimeError("R framework is not available")
        datashape = data.as_matrix().shape
        indepMethod = kwargs.get("indepMethod", "Default")
        # print(datashape, indepMethod)

        args = {"indepTest": kwargs.get("indepTest", RPackages.pcalg.gaussCItest),
                "alpha": kwargs.get("alpha", 0.1),
                "p": datashape[1],
                "fixedGaps": np.zeros((datashape[1],)*2),
                "fixedEdges": np.zeros((datashape[1],)*2),
                "NAdelete": kwargs.get("NAdelete", True),
                "m.max": kwargs.get("mmax", r("Inf")),
                "u2pd": kwargs.get("u2pd", "relaxed"),
                "skel.method": kwargs.get("skelmethod", "stable"),
                "conservative": kwargs.get("conservative", False),
                "maj.rule": kwargs.get("majrule", True),
                "solve.confl": kwargs.get("solveconfl", False),
                "numCores": kwargs.get("nb_jobs", SETTINGS.NB_JOBS),
                "verbose": kwargs.get("verbose", SETTINGS.verbose)
                }

        if indepMethod == "Default":
            # print("OK")
            args["suffStat"] = rlc.TaggedList([r.cov(data.as_matrix()), datashape[0]], tags=("C", "n"))
        else:
            args["suffStat"] = rlc.TaggedList([data.as_matrix(), indepMethod], tags=("data", "ic.method"))
        # print(args, args["fixedGaps"].shape, datashape, np.corrcoef(data.as_matrix(), rowvar=False).shape)
        pcfit = RPackages.pcalg.pc(**args)
        pcmat = np.array(r("as")(pcfit, "matrix"), dtype=int)
        # Removing undirected edges from estimated adjacency matrix
        result = DataFrame(pcmat * np.transpose(1-pcmat), columns=data.columns)

        return DirectedGraph(df=result, adjacency_matrix=True)
