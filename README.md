# Causal Discovery Toolbox

Package for causal inference in graphs and in the pairwise settings. Tools for graph structure recovery and dependencies are included. 

It implements the CGNN algorithm from the paper :
1. [**Learning Functional Causal Models with Generative Neural Networks**](https://arxiv.org/abs/1709.05321), Olivier Goudet, Diviyan Kalainathan, Philippe Caillou, David Lopez-Paz, Isabelle Guyon, Michèle Sebag, Aris Tritas, Paola Tubaro.

The CGNN can be used using :

```python
from cdt.causality.graph import CGNN
output = CGNN.predict(data, skeleton)
```

An example of application on the LUCAS dataset (on Lung cancer) using CGNNs can be found here : [jupyter-notebook](LUCAS_example/Discovery_LUCAS.ipynb)

**A More detailed documentation will be soon released**

## Project structure

[See here](Specifications.md)

## Installation

[Check the installation instructions here](installation_instructions.md)



