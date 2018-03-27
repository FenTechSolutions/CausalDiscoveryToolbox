"""Feature selection model with generative models.

Algorithm between SAM and CGNN
Author : Diviyan Kalainathan & Olivier Goudet
"""
import os
import pandas as pd
import torch as th
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
              train_epochs=1000, test_epochs=1000, gpu=False, gpuno=0):
        optim = th.optim.Adam(self.parameters(), lr=lr)
        output = th.zeros(x.size()[1])
        xtr = Variable(x)
        ytr = Variable(y)
        noise = Variable(th.randn(xtr.size()))
        output = th.zeros(x.size()[1])

        if gpu:
            noise = noise.cuda(gpuno)
            output = output.cuda(gpuno)

        criterion = MMDloss(input_size=x.size()[0], gpu=gpu, gpu_id=gpuno)
        # if batch_size == -1:
        #     batch_size = x.size()[0]
        # Printout value
        # data_iterator = th.utils.data.DataLoader(x, batch_size=batch_size,
        #                                          shuffle=True)
        # TRAIN
        for epoch in range(train_epochs + test_epochs):
            optim.zero_grad()
            gen = self(xtr)
            loss = criterion(gen, ytr) + l1*self.layers[0].abs().sum()
            # Train the discriminator
            if epoch >= train_epochs:
                output.add_(self.layers[0].data.sum(dim=0))
            loss.backward()
            optim.step()

        return list(output.cpu().numpy())


def run_FSGNN(data, target):
    return 0


class FSGNN(FeatureSelectionModel):
    def __init__(self):
        super(FSGNN, self).__init__()
