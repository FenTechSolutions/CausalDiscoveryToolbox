PyTorch Models
==============

In order to have more flexibility in the use of neural network models,
these are directly assessible as `torch.nn.Module`, using the extensions `.model`, for example:

.. code-block:: python

   >>> from cdt.causality.graph.model.CGNN

to import the CGNN Pytorch model. The available models are the following:

- CGNN

- SAM

- NCC

- GNN

- FSGNN

.. currentmodule:: cdt.causality.graph.model

CGNN
----

.. autoclass:: CGNN_model
   :members:

SAM
---

.. autoclass:: SAM_generators
   :members:

.. autoclass:: SAM_discriminator
   :members:

.. currentmodule:: cdt.causality.pairwise.model

NCC
---

.. autoclass:: NCC_model
   :members:

GNN
---

.. autoclass:: GNN_model
   :members:

.. currentmodule:: cdt.independence.graph.model

FSGNN
-----

.. autoclass:: FSGNN_model
   :members:
