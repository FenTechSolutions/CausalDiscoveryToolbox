Advanced Tutorial
=================

This second tutorial targets more experienced users. We will focus on:

1. Launching `cdt` Docker containers

2. Tweaking the ``cdt.SETTINGS`` to adapt the package to the hardware
   configuration

3. Generate a artificial dataset from scratch

4. Perform causal discovery on GPU

5. Evaluate the results


1. Launch the Docker containers
-------------------------------
Docker images are really useful to have a portable environment with minimal
impact on performance. In our case, it becomes really handy as all the R
libraries are quite time-consuming to install and have lots of
incompatibilities depending on the user environment. Check
https://docs.docker.com/install/ to install Docker and have a quick tutorial
on its usage.

`cdt` Docker containers are available at https://hub.docker.com/u/divkal .
Check :ref:`here <Docker images>` to select the image adapted to your
configuration.
In this tutorial we will consider having GPUs available, but the methods are
really similar if you don't have GPUs (selecting the CPU docker image instead
of the GPU one).

.. code-block:: bash

   $ docker pull divkal/nv-cdt-py3.6:XX  # XX corresponds to the latest version
   $ nvidia-docker run -it --init --ipc=host --rm -u=$(id -u):$(id -g) divkal/nv-cdt-py3.6:XX /bin/bash
   =============
   == PyTorch ==
   =============

   NVIDIA Release 18.09 (build 687447)

   Container image Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

   Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
   Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
   Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
   Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
   Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
   Copyright (c) 2011-2013 NYU                      (Clement Farabet)
   Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
   Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
   Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
   All rights reserved.

   Various files include modifications (c) NVIDIA CORPORATION.  All rights reserved.
   NVIDIA modifications are covered by the license terms that apply to the underlying project or file.
   Failed to detect NVIDIA driver version.

   I have no name!@5308f95cd331:/workspace$
   I have no name!@5308f95cd331:/workspace$ ipython
   Python 3.6.5 |Anaconda, Inc.| (default, Apr 29 2018, 16:14:56)
   Type 'copyright', 'credits' or 'license' for more information
   IPython 6.5.0 -- An enhanced Interactive Python. Type '?' for help.

   In [1]:


The docker image is built upon the Nvidia NGC docker image for PyTorch. Details
of the options of the docker command:

- ``nvidia-docker`` is a variant of ``docker`` developed by NVIDIA for GPU
  passthrough. It is available at : https://github.com/NVIDIA/nvidia-docker

- ``-it`` is an option to launch the container in interactive mode

- ``--init`` is to passthrough the signals such as SIGINT or SIGKILL in the
  container.

- ``--rm`` is an option to save space by deleting the container at the end
  of the execution.

- ``-u`` is an option to launch the container as a specific user. Otherwise it
  will be executed as ``root``. This is quite useful for accessing files
  created in the container from the outside environment.

2. Adapt the `cdt` package configuration
----------------------------------------

In this section, we will tweak the ``cdt.SETTINGS`` to fit our usage.
We will first check the current configuration, then increase the number of jobs
as the graph generated in the next section will be quite small. More details
on the package settings are :ref:`provided here <Toolbox Settings>`.


.. code-block:: python

   In [1]: import cdt
   Detecting 1 CUDA device(s).

   In [2]: cdt.SETTINGS.GPU  # Is set to the number of devices
   Out[2]: 1

   In [3]: cdt.SETTINGS.NJOBS  # Set to the num of devices
   Out[3]: 1

   In [4]: cdt.SETTINGS.NJOBS = 3  # 3 jobs per GPU

   In [5]: cdt.SETTINGS.verbose = False

3. Artifical graph generation
-----------------------------

Generating artificial graph with the `cdt` package is quite straightforward when
using the ``cdt.data.AcyclicGraphGenerator`` class. :ref:`Check here
<AcyclicGraphGenerator>` to have more details on how to customize the graph
generator.

.. code-block:: python

   In [6]: generator = cdt.data.AcyclicGraphGenerator('gp_add', noise_coeff=.2,
                                                      nodes=20, parents_max=3)

   In [7]: data, graph = generator.generate()

   In [7]: data.head()
   Out[7]:
            V0        V1        V2        V3    ...          V16       V17       V18       V19
   0 -0.948506  0.366023 -0.659409 -1.012921    ...    -0.086537  0.504257  1.163381 -0.815508
   1 -1.175473  1.612285  1.087017 -1.505346    ...    -0.119292 -1.251204  0.303203 -0.730214
   2 -0.899956  0.757223 -0.394799 -1.345747    ...    -0.620322 -0.919279 -1.948743  0.027883
   3 -1.143217  1.419192  0.608848 -1.144207    ...     1.992465 -1.277411 -0.109563 -0.907268
   4 -0.653106 -0.582684 -0.947306 -0.701014    ...    -0.217655  1.429272 -1.156742  1.305437

   [5 rows x 20 columns]


And the data and graph are generated.

4. Run SAM on GPUs
------------------

Running multiple bootstrapped runs of SAM proved itself to yield much better
results than a single run. The parameter ``nruns`` allows to control the total
number of runs. As soon as the setting ``cdt.SETTINGS.GPU > 0``, the execution
of GPU compatible algorithms will be automatically performed on those devices,
making the prediction step similar to a traditional algorithm:

.. code-block:: python

   In [8]: sam = cdt.causality.graph.SAM(nruns=12)

   In [9]: prediction = sam.predict(data)

.. seealso::

   Kalainathan, Diviyan & Goudet, Olivier & Guyon, Isabelle & Lopez-Paz, David
   & Sebag, Michèle. (2018). SAM: Structural Agnostic Model, Causal Discovery
   and Penalized Adversarial Learning.

5. Scoring the results
----------------------
In a similar fashion to the other tutorial, we can quickly score the results
using the methods in ``cdt.metrics``:

.. code-block:: python

   In [10]: from cdt.metrics import (precision_recall, SHD)

   In [11]: [metric(graph, prediction) for metric in
            (precision_recall, SHD)]
   Out[11]: [(0.53, [(0.06, 1.0), (1.0, 0.0)]), 24.0]

This concludes our second tutorial on the `cdt` package.
