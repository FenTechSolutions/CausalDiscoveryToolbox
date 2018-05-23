"""GNN : Generative Neural Networks for causal inference (pairwise).

Authors : Olivier Goudet & Diviyan Kalainathan
Ref: Causal Generative Neural Networks (https://arxiv.org/abs/1711.08936)
Date : 10/05/2017
"""
import numpy as np
from ...utils.loss import MMDloss, TTestCriterion
from ...utils.Settings import SETTINGS
from joblib import Parallel, delayed
from sklearn.preprocessing import scale
import torch as th
from torch.autograd import Variable
from .model import PairwiseModel


class GNN_model(th.nn.Module):
    """Torch model for the GNN structure."""

    def __init__(self, batch_size, nh=20, device=None):
        """Build the Torch graph.

        :param batch_size: size of the batch going to be fed to the model
        :param kwargs: h_layer_dim=(CGNN_SETTINGS.h_layer_dim)
                       Number of units in the hidden layer
        :param device: device on with the algorithm is going to be run on.
        """
        super(GNN_model, self).__init__()
        device = SETTINGS.get_default(device=device)
        self.l1 = th.nn.Linear(2, nh)
        self.l2 = th.nn.Linear(nh, 1)
        self.noise = Variable(th.FloatTensor(
            batch_size, 1), requires_grad=False).to(device)
        self.act = th.nn.ReLU()
        self.criterion = MMDloss(batch_size, device=device)
        self.layers = th.nn.Sequential(self.l1, self.act, self.l2)

    def forward(self, x):
        """Pass data through the net structure.

        :param x: input data: shape (:,1)
        :type x: torch.Variable
        :return: output of the shallow net
        :rtype: torch.Variable

        """
        self.noise.normal_()
        return self.layers(th.cat([x, self.noise], 1))

    def run(self, x, y, lr=0.01, train_epochs=1000, test_epochs=1000, idx=0, verbose=None, **kwargs):
        """Run the GNN on a pair x,y of FloatTensor data."""
        verbose = SETTINGS.get_default(verbose=verbose)
        optim = th.optim.Adam(self.parameters(), lr=lr)
        running_loss = 0
        teloss = 0

        for i in range(train_epochs + test_epochs):
            optim.zero_grad()
            pred = self.forward(x)
            loss = self.criterion(pred, y)
            running_loss += loss.item()

            if i < train_epochs:
                loss.backward()
                optim.step()
            else:
                teloss += running_loss

            # print statistics
            if verbose and not i % 300:
                print('Idx:{}; epoch:{}; score:{}'.
                      format(idx, i, running_loss/300))
                running_loss = 0.0

        return teloss / test_epochs


def GNN_instance(x, idx=0, device=None, nh=20, **kwargs):
    """Run an instance of GNN, testing causal direction.

    :param m: data corresponding to the config : (N, 2) data, [:, 0] cause and [:, 1] effect
    :param pair_idx: print purposes
    :param run: numner of the run (for GPU dispatch)
    :param device: device on with the algorithm is going to be run on.
    :return:
    """
    device = SETTINGS.get_default(device=device)
    xy = scale(x).astype('float32')
    inputx = th.FloatTensor(xy[:, [0]]).to(device)
    target = th.FloatTensor(xy[:, [1]]).to(device)
    GNNXY = GNN_model(x.shape[0], device=device, nh=nh).to(device)
    GNNYX = GNN_model(x.shape[0], device=device, nh=nh).to(device)

    XY = GNNXY.run(inputx, target, **kwargs)
    YX = GNNYX.run(target, inputx, **kwargs)

    return [XY, YX]


# Test
class GNN(PairwiseModel):
    """Shallow Generative Neural networks.

    Models the causal directions x->y and y->x with a 1-hidden layer neural network
    and a MMD loss. The causal direction is considered as the "best-fit" between the two directions
    """

    def __init__(self, nh=20, lr=0.01):
        """Init the model."""
        super(GNN, self).__init__()
        self.nh = nh
        self.lr = lr

    def predict_proba(self, a, b, nb_runs=6, nb_jobs=None, gpu=None,
                      idx=0, verbose=None, ttest_threshold=0.01,
                      nb_max_runs=16, train_epochs=1000, test_epochs=1000):
        """Run multiple times GNN to estimate the causal direction."""
        nb_jobs, verbose, gpu = SETTINGS.get_default(('nb_jobs', nb_jobs), ('verbose', verbose), ('gpu', gpu))
        x = np.stack([a.ravel(), b.ravel()], 1)
        ttest_criterion = TTestCriterion(
            max_iter=nb_max_runs, runs_per_iter=nb_runs, threshold=ttest_threshold)

        AB = []
        BA = []

        while ttest_criterion.loop(AB, BA):
            if nb_jobs != 1:
                result_pair = Parallel(n_jobs=nb_jobs)(delayed(GNN_instance)(
                    x, idx=idx, device='cuda:{}'.format(run % gpu) if gpu else 'cpu',
                    verbose=verbose, train_epochs=train_epochs, test_epochs=test_epochs) for run in range(ttest_criterion.iter, ttest_criterion.iter + nb_runs))
            else:
                result_pair = [GNN_instance(x, idx=idx,
                                            device='cuda:0' if gpu else 'cpu',
                                            verbose=verbose,
                                            train_epochs=train_epochs,
                                            test_epochs=test_epochs)
                               for run in range(ttest_criterion.iter, ttest_criterion.iter + nb_runs)]
            AB.extend([runpair[0] for runpair in result_pair])
            BA.extend([runpair[1] for runpair in result_pair])

        if verbose:
            print("P-value after {} runs : {}".format(ttest_criterion.iter,
                                                      ttest_criterion.p_value))

        score_AB = np.mean(AB)
        score_BA = np.mean(BA)

        return (score_BA - score_AB) / (score_BA + score_AB)
