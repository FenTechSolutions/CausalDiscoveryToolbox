.. role:: hidden
    :class: hidden-section

===========
Get started
===========

This section focuses on explaining the general functionalities of the package,
and how its components interact with each other. Then two tutorials are
provided:

- :ref:`The first one going through the main features of the package <Basic Tutorial>`

- :ref:`The second for GPU users, highlighting advanced features. <Advanced Tutorial>`

For an installation guide, please check :ref:`here <Installation>`


.. toctree::
   :hidden:

   tutorial_1
   tutorial_2


Package Description
===================

General package architecture
----------------------------
The Causal Discovery Toolbox is a package for causal discovery in the
observational setting. Therefore support for data with interventions is not
available at the moment, but is considered for later versions.

The package is structured in 5 modules:

1. Causality: ``cdt.causality`` implements algorithms for causal discovery, either
   in the pairwise setting or the graph setting.

2. Independence: ``cdt.independence`` includes methods to recover the dependence
   graph of the data.

3. Data: ``cdt.data`` provides the user with tools to generate data, and load
   benchmark data.

4. Utils: ``cdt.utils`` provides tools to the users for model
   construction, graph utilities and settings.

5. Metrics: ``cdt.metrics`` includes scoring metrics for graphs, taking as input
   ``networkx.DiGraph``

All methods for computation adopt a 'scikit-learn' like interface, where ``.predict()``
manages to launch the algorithm on the given data to the toolbox, ``.fit()`` allows
to train learning algorithms  Most of the algorithms are classes, and their
parameters can be customized in the ``.__init__()`` function of the class.

.. note::
   The ``.predict()`` function is often implemented in the base class
   (``cdt.causality.graph.GraphModel`` for causal graph algorithms).
   ``.predict()`` is often a wrapper calling sub-functions depending on the
   arguments fed to the functions. The sub-functions, such as
   ``.orient_directed_graph()`` for ``cdt.causality.graph`` models (which is
   called when a directed graph is fed as a second argument ), are
   implemented and documented in the various algorithms.


Hardware and algorithm settings
-------------------------------
The toolbox has a SETTINGS class that defines the hardware settings. Those
settings are unique and their default parameters are defined in
``cdt/utils/Settings``.

These parameters are accessible and overridable via accessing the class:

.. code-block:: python

   >>> import cdt
   >>> cdt.SETTINGS


Moreover, the hardware parameters are detected and defined automatically
(including number of GPUs, CPUs, available optional packages) at the ``import``
of the package using the ``cdt.utils.Settings.autoset_settings`` method, ran at
startup.

These settings are overriddable in two ways:

1. By changing ``cdt.SETTINGS`` attributes, thus changing the SETTINGS for the
   whole python session.

2. By changing the parameters of the functions/classes used. When their default
   value in the class definition is ``None``, the ``cdt.SETTINGS`` value is taken, by
   using the ``cdt.SETTINGS.get_default`` function. This allows for quick and
   temporary parameter change.

The graph class
---------------
The whole package revolves around using the graph classes of the ``networkx``
package.
Most of the methods have the option of predicting a directed graph
(`networkx.DiGraph`) or an undirected graph (`networkx.Graph`).

The ``networkx`` library might not be intuitive to use at first, but it comes
with many useful tools for graphs. Here is a list of handy function, to
understand the output of the toolbox's outputs:

.. code-block:: python

   >>> import networkx as nx
   >>> g = nx.DiGraph()  # initialize a directed graph
   >>> l = list(g.nodes())  # list of nodes in the graph
   >>> a = nx.adj_matrix(g).todense()  # Output the adjacency matrix of the graph
   >>> e = list(g.edges())  # list of edges in the graph

Please refer to `networkx` 's documentation for more detailed information:
https://https://networkx.github.io/documentation/stable/

