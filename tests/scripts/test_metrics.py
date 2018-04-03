"""Test the metrics."""
import numpy as np
import networkx as nx
from cdt.utils.metrics import precision_recall, SHD, SID


def init():
    mat1 = np.zeros((3, 3))
    mat1[1, 0] = 1
    mat1[2, 1] = 1
    mat1[2, 0] = 1
    mat2 = np.matrix.copy(mat1)
    mat2[2, 1] = 0
    mat2[1, 2] = .5
    return nx.DiGraph(mat1), nx.DiGraph(mat2)


def test_precision_recall():
    assert precision_recall(*init())


def test_SHD():
    assert SHD(*init()) == 2
    assert SHD(*init(), double_for_anticausal=False) == 1


def test_SID():
    assert SID(*init()) == 4.0


if __name__ == '__main__':
    a, b = init()
    print(nx.adj_matrix(a), nx.adj_matrix(b))
    print(SID(a, b))
