""" This module focusses on data: data generation but also provides the user
with standard and well known datasets, useful for validation and benchmarking.

The generators provide the user the ability to choose which causal mechanism to
be used in the data generation process, as well as the type of noise
contribution (additive and/or multiplicative). Currently, the implemented
mechanisms are (:math:`+\\times` denotes either addition or multiplication, and
:math:`\\mathbf{X}` denotes the vector of causes, and :math:`E` represents the
noise variable accounting for all unobserved variables):

- Linear: :math:`y = \\mathbf{X}W +\\times E`
- Polynomial: :math:`y = \\left( W_0 + \\mathbf{X}W_1 + ...+ \\mathbf{X}^d W_d \\right) +\\times E`
- Gaussian Process: :math:`y = GP(\\mathbf{X}) +\\times E`
- Sigmoid: :math:`y = \\sum_i^d W_i * sigmoid(\\mathbf{X_i}) +\\times E`
- Randomly init. Neural network: :math:`y = \\sigma((\\mathbf{X},E) W_{in})W_{out}`

Causal pairs can be generated using the ``cdt.data.CausalPairGenerator`` class,
and acyclic graphs can be generated using the ``cdt.data.AcyclicGraphGenerator`` class.


.. MIT License
..
.. Copyright (c) 2018 Diviyan Kalainathan
..
.. Permission is hereby granted, free of charge, to any person obtaining a copy
.. of this software and associated documentation files (the "Software"), to deal
.. in the Software without restriction, including without limitation the rights
.. to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
.. copies of the Software, and to permit persons to whom the Software is
.. furnished to do so, subject to the following conditions:
..
.. The above copyright notice and this permission notice shall be included in all
.. copies or substantial portions of the Software.
..
.. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
.. IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
.. FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
.. AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
.. LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
.. OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
.. SOFTWARE.
"""

from .acyclic_graph_generator import AcyclicGraphGenerator
from .causal_pair_generator import CausalPairGenerator
from .loader import load_dataset
