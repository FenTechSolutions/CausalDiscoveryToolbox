"""Feature selection model with generative models.

Algorithm between SAM and CGNN
Author : Diviyan Kalainathan & Olivier Goudet

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
import torch as th
import numpy as np
from torch.utils import data
from sklearn.preprocessing import scale
from tqdm import trange
from .model import FeatureSelectionModel
from ...utils.Settings import SETTINGS
from ...utils.loss import MMDloss


class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, dataset, labels):
        'Initialization'
        self.labels = labels
        self.dataset = dataset

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        # Load data and get label

        return self.dataset[index], self.labels[index]


class FSGNN_model(th.nn.Module):
    def __init__(self, sizes, dropout=0., activation_function=th.nn.ReLU):
        super(FSGNN_model, self).__init__()
        layers = []

        for i, j in zip(sizes[:-2], sizes[1:-1]):
            layers.append(th.nn.Linear(i, j))
            if dropout != 0.:
                layers.append(th.nn.Dropout(p=dropout))
            layers.append(activation_function())

        layers.append(th.nn.Linear(sizes[-2], sizes[-1]))
        self.layers = th.nn.Sequential(*layers)

    def forward(self, x):
        self.layers(x)

    def train(self, x, y, lr=0.01, l1=0.1, batch_size=-1,
              train_epochs=1000, test_epochs=1000, device=None,
              verbose=None):
        device, verbose = SETTINGS.get_default(('device', device), ('verbose', verbose))
        optim = th.optim.Adam(self.parameters(), lr=lr)
        output = th.zeros(x.size()[1])
        output = th.zeros(x.size()[1]).to(device)

        criterion = MMDloss(input_size=x.shape[0], device=device)
        if batch_size == -1:
            batch_size = x.size()[0]
        # Printout value
        noise = th.randn(batch_size, 1).to(device)
        data_iterator = th.utils.data.DataLoader(Dataset(x, y), batch_size=batch_size,
                                                 shuffle=True, drop_last=True)
        # TRAIN
        with trange(train_epochs + test_epochs, disable=not verbose) as t:
            for epoch in t:
                for i, (batch, target) in enumerate(data_iterator):
                    optim.zero_grad()
                    noise.normal_()
                    gen = self.layers(th.cat([batch, noise], 1))
                    # print(gen)
                    loss = criterion(gen, target) + l1*(self.layers[0].weight.abs().sum() + self.layers[2].weight.abs().sum())
                    # Train the discriminator
                    if not epoch % 100 and i == 0:
                        t.set_postfix(epoch=epoch, loss=loss.item())
                    if epoch >= train_epochs:
                        output.add_(self.layers[0].weight.data[:, :-1].sum(dim=0))
                    loss.backward()
                    optim.step()

        return list(output.div_(test_epochs).div_(x.shape[0]//batch_size).cpu().numpy())


class FSGNN(FeatureSelectionModel):
    """Feature Selection using MMD and Generative Neural Networks."""

    def __init__(self):
        """Init the model."""
        super(FSGNN, self).__init__()

    def predict_features(self, df_features, df_target, nh=20, idx=0, dropout=0.,
                         activation_function=th.nn.ReLU, lr=0.01, l1=0.1,  batch_size=-1,
                         train_epochs=1000, test_epochs=1000, device=None,
                         verbose=None, nb_runs=3):
        """For one variable, predict its neighbours.

        Args:
            df_features (pandas.DataFrame):
            df_target (pandas.Series):
            nh (int): number of hidden units
            idx (int): (optional) for printing purposes
            dropout (float): probability of dropout (between 0 and 1)
            activation_function (torch.nn.Module): activation function of the NN
            lr (float): learning rate of Adam
            l1 (float): L1 penalization coefficient
            batch_size (int): batch size, defaults to full-batch
            train_epochs (int): number of train epochs
            test_epochs (int): number of test epochs
            device (str): cuda or cpu device (defaults to ``cdt.SETTINGS.default_device``)
            verbose (bool): verbosity (defaults to ``cdt.SETTINGS.verbose``)
            nb_runs (int): number of bootstrap runs

        Returns:
            list: scores of each feature relatively to the target

        """
        device, verbose = SETTINGS.get_default(('device', device), ('verbose', verbose))
        x = th.FloatTensor(scale(df_features.values)).to(device)
        y = th.FloatTensor(scale(df_target.values)).to(device)
        out = []
        for i in range(nb_runs):
            model = FSGNN_model([x.size()[1] + 1, nh, 1],
                                dropout=dropout,
                                activation_function=activation_function).to(device)

            out.append(model.train(x, y, lr=0.01, l1=0.1, batch_size=-1,
                                   train_epochs=train_epochs, test_epochs=test_epochs,
                                   device=device, verbose=verbose))
        return list(np.mean(np.array(out), axis=0))
