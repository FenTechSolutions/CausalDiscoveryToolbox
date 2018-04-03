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

    def __init__(self, batch_size, nh=20, gpu=SETTINGS.GPU, gpu_id=0):
        """Build the Torch graph.

        :param batch_size: size of the batch going to be fed to the model
        :param kwargs: h_layer_dim=(CGNN_SETTINGS.h_layer_dim)
                       Number of units in the hidden layer
        :param kwargs: gpu=(SETTINGS.GPU), if GPU is used for computations
        :param kwargs: gpu_no=(0), GPU ID
        """
        super(GNN_model, self).__init__()
        self.l1 = th.nn.Linear(2, nh)
        self.l2 = th.nn.Linear(nh, 1)
        self.noise = Variable(th.FloatTensor(
            batch_size, 1), requires_grad=False)
        if gpu:
            self.noise = self.noise.cuda(gpu_id)
        self.act = th.nn.ReLU()
        self.criterion = MMDloss(batch_size, gpu=gpu, gpu_id=gpu_id)

    def forward(self, x):
        """Pass data through the net structure.

        :param x: input data: shape (:,1)
        :type x: torch.Variable
        :return: output of the shallow net
        :rtype: torch.Variable

        """
        self.noise.normal_()
        y = self.act(self.l1(th.cat([x, self.noise], 1)))
        return self.l2(y)

    def run(self, x, y, lr=0.01, train_epochs=1000, test_epochs=1000, idx=0):
        """Run the GNN on a pair x,y of FloatTensor data."""
        optim = th.optim.Adam(self.parameters(), lr=lr)
        running_loss = 0
        teloss = 0

        for i in range(train_epochs + test_epochs):
            optim.zero_grad()
            pred = self(x)
            loss = self.criterion(pred, y)
            running_loss += loss.data[0]

            if i < train_epochs:
                loss.backward()
                optim.step()
            else:
                teloss += running_loss

            # print statistics
            if not i % 300:
                print('Idx:{} ; score:{}'.
                      format(idx, running_loss))
                running_loss = 0.0

        return teloss / test_epochs


def GNN_instance(x, idx=0, gpu=SETTINGS.GPU, gpu_id=0, **kwargs):
    """Run an instance of GNN, testing causal direction.

    :param m: data corresponding to the config : (N, 2) data, [:, 0] cause and [:, 1] effect
    :param pair_idx: print purposes
    :param run: numner of the run (for GPU dispatch)
    :param kwargs: gpu=(SETTINGS.GPU) True if GPU is used
    :param kwargs: nb_gpu=(SETTINGS.NB_GPU) Number of available GPUs
    :param kwargs: gpu_offset=(SETTINGS.GPU_OFFSET) number of gpu offsets
    :return:
    """
    xy = scale(x).astype('float32')
    inputx = Variable(th.FloatTensor(xy[:, 0]))
    target = Variable(th.FloatTensor(xy[:, 1]))
    GNNXY = GNN_model(x.shape[0], gpu=gpu, gpu_id=gpu_id, **kwargs)
    GNNYX = GNN_model(x.shape[0], gpu=gpu, gpu_id=gpu_id, **kwargs)
    if gpu:
        target = target.cuda(gpu_id)
        inputx = inputx.cuda(gpu_id)
        GNNXY = GNNXY.cuda(gpu_id)
        GNNYX = GNNYX.cuda(gpu_id)
    XY = GNNXY.run(inputx, target, **kwargs)
    YX = GNNYX.run(target, inputx, **kwargs)

    return [XY, YX]


# Test
class GNN(PairwiseModel):
    """Shallow Generative Neural networks.

    Models the causal directions x->y and y->x with a 1-hidden layer neural network
    and a MMD loss. The causal direction is considered as the "best-fit" between the two directions
    """

    def __init__(self):
        """Init the model."""
        super(GNN, self).__init__()

    def predict_proba(self, a, b, nb_runs=6, nb_jobs=SETTINGS.NB_JOBS,
                      idx=0, verbose=SETTINGS.verbose, ttest_threshold=0.01,
                      nb_max_runs=16, **kwargs):
        """Run multiple times GNN to estimate the causal direction."""
        x = np.concatenate([a, b], 1)
        ttest_criterion = TTestCriterion(
            max_iter=nb_max_runs, runs_per_iter=nb_runs, threshold=ttest_threshold)

        AB = []
        BA = []

        while ttest_criterion.loop(AB, BA):
            result_pair = Parallel(n_jobs=nb_jobs)(delayed()(
                x, idx=idx, **kwargs) for run in range(ttest_criterion.iter, ttest_criterion.iter + nb_runs))
            AB.extend([runpair[0] for runpair in result_pair])
            BA.extend([runpair[1] for runpair in result_pair])

        if verbose:
            print("P-value after {} runs : {}".format(ttest_criterion.iter,
                                                      ttest_criterion.p_value))

        score_AB = np.mean(AB)
        score_BA = np.mean(BA)

        return (score_BA - score_AB) / (score_BA + score_AB)


if __name__ == "__main__":
    print("Testing GNN..")
    raise NotImplementedError
