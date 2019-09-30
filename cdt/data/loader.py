""" Dataset loading utilities.

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
import pandas as pd
import os
import requests
import zipfile
import io
from numpy import random
from ..utils.io import read_causal_pairs, read_list_edges


def load_dataset(name, **kwargs):
    """Main function of this module, allows to easily import well-known causal
    datasets into python.

    Details on the supported datasets:
        + **tuebingen**, dataset of 100 real cause-effect pairs
           J. M. Mooij,
           J. Peters, D. Janzing, J. Zscheischler, B. Schoelkopf: "Distinguishing
           cause from effect using observational data: methods and benchmarks",
           Journal of Machine Learning Research 17(32):1-102, 2016.

        + **sachs**, Dataset of flow cytometry, real data,
           11 variables x 7466
           samples; Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A., & Nolan,
           G. P. (2005). Causal protein-signaling networks derived from
           multiparameter single-cell data. Science, 308(5721), 523-529.

        + **dream4**, multifactorial artificial data of the challenge.
           Data generated with GeneNetWeaver 2.0, 5 graphs of 100 variables x 100
           samples. Marbach D, Prill RJ, Schaffter T, Mattiussi C, Floreano D,
           and Stolovitzky G. Revealing strengths and weaknesses of methods for
           gene network inference. PNAS, 107(14):6286-6291, 2010.

    Args:
        name (str): Name of the dataset. currenly supported datasets:
           [`tuebingen`, `sachs`, `dream4-1`, `dream4-2`, `dream4-3`,
           `dream4-4`, `dream4-5`]
        \**kwargs: Optional additional arguments for dataset loaders.
           ``tuebingen`` dataset accepts the ``shuffle (bool)`` option to
           shuffle the causal pairs and their according labels.

    Returns:
        tuple: (pandas.DataFrame, pandas.DataFrame or networkx.DiGraph) Standard
        dataframe containing the data, and the target.

    Examples:
        >>> from cdt.data import load_dataset
        >>> s_data, s_graph = load_dataset('sachs')
        >>> t_data, t_labels = load_dataset('tuebingen')

    .. warning::
       The 'Tuebingen' dataset is loaded with the same label for all samples (1: A causes B)
    """
    dream = [i for i in ['dream4-{}'.format(v) for v in range(1, 6)]]
    loaders = {'tuebingen': load_tuebingen, 'sachs': load_sachs}
    loaders.update({i: load_dream_multifactorial(i[-1]) for i in dream})
    try:
        return loaders[name]()

    except KeyError:
        raise ValueError("Unknown dataset name.")


def load_sachs(**kwargs):
    dirname = os.path.dirname(os.path.realpath(__file__))
    return (pd.read_csv('{}/resources/cyto_full_data.csv'.format(dirname)),
            read_list_edges('{}/resources/cyto_full_target.csv'.format(dirname)))


def load_tuebingen(shuffle=False):
    dirname = os.path.dirname(os.path.realpath(__file__))

    data = read_causal_pairs('{}/resources/Tuebingen_pairs.csv'.format(dirname), scale=False)
    labels = pd.read_csv('{}/resources/Tuebingen_targets.csv'.format(dirname)).set_index('SampleID')

    if shuffle:
        for i in range(len(data)):
            if random.choice([True, False]):
                labels.iloc[i, 0] = -1
                buffer = data.iloc[i, 0]
                data.iloc[i, 0] = data.iloc[i, 1]
                data.iloc[i, 1] = buffer

    return data, labels


def load_dream_multifactorial(num, **kwargs):
    idx = num

    def load_d():
        dirname = os.path.dirname(os.path.realpath(__file__))
        target = read_list_edges('{}/resources/goldstandard_insilico100_multifactorial_{}.csv'.format(dirname, idx))
        data = requests.get('https://www.synapse.org/Portal/filehandle?ownerId=syn3049712&ownerType=ENTITY&fileName=DREAM4_InSilico_Size100_Multifactorial.zip&preview=false&wikiId=74630')
        with zipfile.ZipFile(io.BytesIO(data.content)) as f:
            for name in f.namelist():
                if name == 'insilico_size100_{}_multifactorial.tsv'.format(idx):
                    return pd.read_csv(f.open(name), sep='\t'), target
    return load_d
