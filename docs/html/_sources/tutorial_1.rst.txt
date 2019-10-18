Basic Tutorial
==============

In this first tutorial, we will got through all the main features of the `cdt`
package:

1. Loading a dataset

2. Recovering a graph skeleton with independence tests

3. Apply a causal Discovery algorithm

4. Evaluate our approach using 3 different scoring metrics


1. Load data
------------

Loading a standard dataset using the `cdt` package is straightforward using the
``cdt.data`` module. In this
tutorial, we are going to load the `Sachs` dataset.

.. seealso:: `Sachs, K., Perez, O.,
   Pe’er, D., Lauffenburger, D. A., & Nolan, G. P. (2005).
   Causal protein-signaling networks derived from multiparameter single-cell data.
   Science, 308(5721), 523-529`.

This dataset is quite useful as it is quite a small dataset with a relatively
known ground truth and real data.


.. code-block:: python

   >>> import cdt
   >>> data, graph = cdt.data.load_dataset('sachs')
   >>> print(data.head())
   praf  pmek   plcg   PIP2   PIP3  p44/42  pakts473    PKA    PKC   P38  pjnk
   0  26.4  13.2   8.82  18.30  58.80    6.61      17.0  414.0  17.00  44.9  40.0
   1  35.9  16.5  12.30  16.80   8.13   18.60      32.5  352.0   3.37  16.5  61.5
   2  59.4  44.1  14.60  10.20  13.00   14.90      32.5  403.0  11.40  31.9  19.5
   3  73.0  82.8  23.10  13.50   1.29    5.83      11.8  528.0  13.70  28.6  23.1
   4  33.7  19.8   5.19   9.73  24.80   21.10      46.1  305.0   4.66  25.7  81.3

And graph is loaded: the ``data`` object is a ``pandas.DataFrame`` containing all
the data, and ``graph`` contains the ground truth of the dataset:



2. Graph skeleton
-----------------

Having a graph skeleton on given data might be quite useful for having
information on the structure of the data. In order to do so, let's
apply the Graph Lasso.

.. seealso:: `Friedman, J., Hastie, T., & Tibshirani, R. (2008).
   Sparse inverse covariance estimation with the graphical lasso. Biostatistics,
   9(3), 432-441`:

.. code-block:: python

   >>> glasso = cdt.independence.graph.Glasso()
   >>> skeleton = glasso.predict(data)
   >>> print(skeleton)
   <networkx.classes.digraph.DiGraph at 0x7fe3ccfb1438>

The ``skeleton`` object is a ``networkx.Graph`` instance, which might be quite
obscure at first but is handy in the long run. (Check
:ref:`here <The graph class>`  for a quick introduction on ``networkx`` graphs).
We can check the structure of the skeleton by looking at its adjacency matrix:

.. code-block:: python

   >>> print(nx.adjacency_matrix(skeleton).todense())
   matrix([[ 9.26744031e-04, -6.13751618e-04,  1.66612981e-05,
            -1.10912131e-06, -3.04172363e-05, -9.71526466e-05,
             7.00340545e-05, -1.93863471e-06, -7.31774543e-06,
             2.29788237e-06,  8.31264711e-06],
           [-6.13751618e-04,  4.14978956e-04, -1.37962487e-05,
             1.42164753e-06,  2.04443539e-05,  8.24208108e-05,
            -5.63238668e-05,  1.62688021e-06,  8.63444133e-06,
            -2.88779755e-06, -4.69605195e-06],
           [ 1.66612981e-05, -1.37962487e-05,  2.60824802e-04,
            -1.35895911e-04,  8.78979413e-05,  2.17234579e-05,
            -2.07856535e-05,  1.23313600e-06,  2.12954874e-05,
            -3.22869246e-06, -7.47522248e-06],
           [-1.10912131e-06,  1.42164753e-06, -1.35895911e-04,
             8.68622146e-05, -7.05405720e-05,  3.08709259e-06,
            -2.60810094e-06,  9.09261370e-09, -6.25320515e-06,
             2.56399675e-07,  2.85201875e-07],
           [-3.04172363e-05,  2.04443539e-05,  8.78979413e-05,
            -7.05405720e-05,  6.09681818e-04, -9.91703900e-06,
             1.78188074e-05, -5.97491176e-07,  6.11896719e-06,
            -4.30918870e-07,  5.79322379e-06],
           [-9.71526466e-05,  8.24208108e-05,  2.17234579e-05,
             3.08709259e-06, -9.91703900e-06,  1.10860610e-03,
            -3.08483289e-04, -1.30867663e-05, -3.31258890e-05,
             7.76132824e-06,  2.10416319e-05],
           [ 7.00340545e-05, -5.63238668e-05, -2.07856535e-05,
            -2.60810094e-06,  1.78188074e-05, -3.08483289e-04,
             1.66144775e-04,  1.26667898e-06,  3.11407736e-05,
            -7.29116898e-06, -1.86454298e-05],
           [-1.93863471e-06,  1.62688021e-06,  1.23313600e-06,
             9.09261370e-09, -5.97491176e-07, -1.30867663e-05,
             1.26667898e-06,  2.80073467e-06, -3.78879972e-06,
             8.67580852e-07,  6.92379671e-07],
           [-7.31774543e-06,  8.63444133e-06,  2.12954874e-05,
            -6.25320515e-06,  6.11896719e-06, -3.31258890e-05,
             3.11407736e-05, -3.78879972e-06,  1.59642510e-03,
            -2.58155157e-04, -1.01767664e-04],
           [ 2.29788237e-06, -2.88779755e-06, -3.22869246e-06,
             2.56399675e-07, -4.30918870e-07,  7.76132824e-06,
            -7.29116898e-06,  8.67580852e-07, -2.58155157e-04,
             5.32997159e-05, -3.35285721e-06],
           [ 8.31264711e-06, -4.69605195e-06, -7.47522248e-06,
             2.85201875e-07,  5.79322379e-06,  2.10416319e-05,
            -1.86454298e-05,  6.92379671e-07, -1.01767664e-04,
            -3.35285721e-06,  7.05796078e-05]])

As you have noticed already, the graph is quite dense. Let's remove indirect
links in the graph using the Aracne algorithm

.. seealso:: `An Algorithm for the
   Reconstruction of Gene Regulatory Networks in a Mammalian Cellular Context
   Adam A Margolin, Ilya Nemenman, Katia Basso, Chris Wiggins, Gustavo Stolovitzky,
   Riccardo Dalla Favera and Andrea Califano
   DOI: https://doi.org/10.1186/1471-2105-7-S1-S7`

.. code-block:: python

   >>> new_skeleton = cdt.utils.graph.remove_indirect_links(skeleton, alg='aracne')
   >>> print(nx.adjacency_matrix(new_skeleton).todense())
   matrix([[9.26576364e-04, 0.00000000e+00, 1.66279016e-05, 0.00000000e+00,
   0.00000000e+00, 0.00000000e+00, 6.99676073e-05, 0.00000000e+00,
   0.00000000e+00, 2.26182196e-06, 8.29822467e-06],
   [0.00000000e+00, 4.14897907e-04, 0.00000000e+00, 0.00000000e+00,
   2.04095344e-05, 8.22844158e-05, 0.00000000e+00, 1.62835373e-06,
   8.48527014e-06, 0.00000000e+00, 0.00000000e+00],
   [1.66279016e-05, 0.00000000e+00, 2.60808178e-04, 0.00000000e+00,
   8.78753504e-05, 2.17299267e-05, 0.00000000e+00, 1.23340219e-06,
   2.12217433e-05, 0.00000000e+00, 0.00000000e+00],
   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 8.68568259e-05,
   0.00000000e+00, 3.07901285e-06, 0.00000000e+00, 8.94955462e-09,
   0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
   [0.00000000e+00, 2.04095344e-05, 8.78753504e-05, 0.00000000e+00,
   6.09654932e-04, 0.00000000e+00, 1.77984674e-05, 0.00000000e+00,
   0.00000000e+00, 0.00000000e+00, 5.80118715e-06],
   [0.00000000e+00, 8.22844158e-05, 2.17299267e-05, 3.07901285e-06,
   0.00000000e+00, 1.10847276e-03, 0.00000000e+00, 0.00000000e+00,
   0.00000000e+00, 7.72649753e-06, 2.10224309e-05],
   [6.99676073e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
   1.77984674e-05, 0.00000000e+00, 1.66117739e-04, 1.26646124e-06,
   3.10736844e-05, 0.00000000e+00, 0.00000000e+00],
   [0.00000000e+00, 1.62835373e-06, 1.23340219e-06, 8.94955462e-09,
   0.00000000e+00, 0.00000000e+00, 1.26646124e-06, 2.80075082e-06,
   0.00000000e+00, 8.67949681e-07, 6.92548597e-07],
   [0.00000000e+00, 8.48527014e-06, 2.12217433e-05, 0.00000000e+00,
   0.00000000e+00, 0.00000000e+00, 3.10736844e-05, 0.00000000e+00,
   1.59628546e-03, 0.00000000e+00, 0.00000000e+00],
   [2.26182196e-06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
   0.00000000e+00, 7.72649753e-06, 0.00000000e+00, 8.67949681e-07,
   0.00000000e+00, 5.32959890e-05, 0.00000000e+00],
   [8.29822467e-06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
   5.80118715e-06, 2.10224309e-05, 0.00000000e+00, 6.92548597e-07,
   0.00000000e+00, 0.00000000e+00, 7.05766621e-05]])

and the resulting skeleton is much more sparse. Let's build on this new
skeleton to perform our causal discovery.


3. Causal discovery
-------------------

Having a graph skeleton, we are going to perform causal discovery with
constraints, by using the `GES` algorithm.

.. seealso:: `D.M. Chickering (2002). Optimal
   structure identification with greedy search. Journal of Machine Learning
   Research 3 , 507–554`

.. code-block:: python

   >>> model = cdt.causality.graph.GES()
   >>> output_graph = model.predict(data, new_skeleton)
   >>> print(nx.adjacency_matrix(output_graph).todense())
   matrix([[0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
           [1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
           [0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int64)

And we obtain GES's prediction on this graph using the skeleton as constraint.
Next we are going to evaluate our solution compared to using CAM without
skeleton.

.. seealso:: `J. Peters, J.
   Mooij, D. Janzing, B. Schölkopf: Causal Discovery with Continuous Additive Noise
   Models, JMLR 15:2009-2053, 2014.`

4. Evaluation and scoring metrics
---------------------------------

In order to evaluate various predictions with the ground truth, the `cdt`
package includes 3 different metrics in the ``cdt.metrics`` module:

- Area under the precision recall curve

- Structural Hamming Distance (SHD)

- Structural Intervention Distance (SID)

.. code-block:: python

   >>> from cdt.metrics import (precision_recall, SID, SHD)
   >>> scores = [metric(graph, output_graph) for metric in (precision_recall, SID, SHD)]
   >>> print(scores)
   [(0.3212943387361992, [(0.1487603305785124, 1.0), (0.16279069767441862, 0.3888888888888889), (1.0, 0.0)]),
   array(76.),
   47]

   >>> # now we compute the CAM graph without constraints and the associated scores
   >>> model2 = cdt.causality.graph.CAM()
   >>> output_graph_nc = model2.predict(data)
   >>> scores_nc = [metric(graph, output_graph_nc) for metric in (precision_recall, SID, SHD)]
   >>> print(scores_nc)
   [(0.4423624964377315, [(0.1487603305785124, 1.0), (0.3103448275862069, 0.5), (1.0, 0.0)]),
   array(68.),
   29]

We can observe that CAM has better performance than our previous pipeline, as:

- The average precision score is higher

- The SID score is lower

- The SHD score is lower as well

This concludes our first tutorial on the `cdt` package.
