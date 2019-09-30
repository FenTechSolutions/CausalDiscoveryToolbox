"""Test the metrics."""
import numpy as np
import networkx as nx
from cdt.metrics import precision_recall, SHD, SID, SHD_CPDAG, SID_CPDAG
from copy import deepcopy


def init():
    mat1 = np.zeros((3, 3))
    mat1[1, 0] = 1
    mat1[2, 1] = 1
    mat1[2, 0] = 1
    mat2 = deepcopy(mat1)
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

def test_SIDC():
    assert type(SID_CPDAG(*init())) == tuple


def test_SHDC():
    assert type(SHD_CPDAG(*init())) == np.float64

if __name__ == '__main__':
    a, b = init()
    print(SID_CPDAG(a, b))
    print(type(SHD_CPDAG(a, b)) == np.float64)
