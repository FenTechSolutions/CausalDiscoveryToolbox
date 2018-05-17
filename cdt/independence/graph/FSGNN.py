"""Feature selection model with generative models.

Algorithm between SAM and CGNN
Author : Diviyan Kalainathan & Olivier Goudet
"""
import os
import pandas as pd
import torch as th
import numpy as np
from torch.autograd import Variable
from sklearn.preprocessing import scale
from .model import FeatureSelectionModel
from ...utils.Settings import SETTINGS
from ...utils.loss import MMDloss


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

    def train(self, x, y, lr=0.01, l1=0.1,  # batch_size=-1,
              train_epochs=1000, test_epochs=1000, device=None,
              verbose=None):
        device, verbose = SETTINGS.get_default(('device', device), ('verbose', verbose))
        optim = th.optim.Adam(self.parameters(), lr=lr)
        output = th.zeros(x.size()[1])
        noise = Variable(th.randn(x.size())).to(device)
        output = th.zeros(x.size()[1]).to(device)

        criterion = MMDloss(input_size=x.shape[0], device=device)
        # if batch_size == -1:raise NotImplementedError
        #     batch_size = x.size()[0]
        # Printout value
        # data_iterator = th.utils.data.DataLoader(x, batch_size=batch_size,
        #                                          shuffle=True)
        # TRAIN
        for epoch in range(train_epochs + test_epochs):
            optim.zero_grad()
            gen = self.layers(x)
            # print(gen)
            loss = criterion(gen, y) + l1*(self.layers[0].weight.abs().sum() + self.layers[2].weight.abs().sum())
            # Train the discriminator
            if verbose and not epoch % 200:
                print("Epoch: {} ; Loss: {}".format(epoch, loss.item()))
            if epoch >= train_epochs:
                output.add_(self.layers[0].weight.data.sum(dim=0))
            loss.backward()
            optim.step()

        return list(output.div_(test_epochs).cpu().numpy())


class FSGNN(FeatureSelectionModel):
    """Feature Selection using MMD and Generative Neural Networks."""

    def __init__(self):
        """Init the model."""
        super(FSGNN, self).__init__()

    def predict_features(self, df_features, df_target, nh=20, idx=0, dropout=0.,
                         activation_function=th.nn.ReLU, lr=0.01, l1=0.1,  # batch_size=-1,
                         train_epochs=1000, test_epochs=1000, device=None,
                         verbose=None, nb_runs=3):
        """For one variable, predict its neighbours."""
        device, verbose = SETTINGS.get_default(('device', device), ('verbose', verbose))
        x = th.FloatTensor(scale(df_features.as_matrix())).to(device)
        y = th.FloatTensor(scale(df_target.as_matrix())).to(device)
        out = []
        for i in range(nb_runs):
            model = FSGNN_model([x.size()[1], nh, 1],
                                dropout=dropout,
                                activation_function=activation_function).to(device)

            out.append(model.train(x, y, lr=0.01, l1=0.1,  # batch_size=-1,
                                   train_epochs=train_epochs, test_epochs=test_epochs,
                                   device=device, verbose=verbose))
        return list(np.mean(np.array(out), axis=0))