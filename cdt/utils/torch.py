"""PyTorch utilities for models.

Author: Diviyan Kalainathan, Olivier Goudet
Date: 09/3/2018

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
import math
import torch as th
from torch.nn import Parameter
from torch.nn.modules.batchnorm import _BatchNorm
import torch.distributions.relaxed_bernoulli as relaxed_bernoulli
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import SigmoidTransform,AffineTransform
from torch.distributions.uniform import Uniform


def _sample_gumbel(shape, eps=1e-10, out=None):
    """
    Implementation of pytorch.
    (https://github.com/pytorch/pytorch/blob/e4eee7c2cf43f4edba7a14687ad59d3ed61d9833/torch/nn/functional.py)
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else th.rand(shape)
    return - th.log(eps - th.log(U + eps))


def _gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    Implementation of pytorch.
    (https://github.com/pytorch/pytorch/blob/e4eee7c2cf43f4edba7a14687ad59d3ed61d9833/torch/nn/functional.py)
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    dims = logits.dim()
    gumbel_noise = _sample_gumbel(logits.size(), eps=eps, out=logits.data.new())
    y = logits + gumbel_noise
    return th.softmax(y / tau, dims-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    Implementation of pytorch.
    (https://github.com/pytorch/pytorch/blob/e4eee7c2cf43f4edba7a14687ad59d3ed61d9833/torch/nn/functional.py)
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: `[batch_size, n_class]` unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if ``True``, take `argmax`, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes
    Constraints:
    - this implementation only works on batch_size x num_features tensor for now
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    shape = logits.size()
    assert len(shape) == 2
    y_soft = _gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = logits.data.new(*shape).zero_().scatter_(-1, k.view(-1, 1), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = y_hard - y_soft.data + y_soft
    else:
        y = y_soft
    return y


def _sample_logistic(shape, out=None):

    U = out.resize_(shape).uniform_() if out is not None else th.rand(shape)
    #U2 = out.resize_(shape).uniform_() if out is not None else th.rand(shape)
    return th.log(U) - th.log(1-U)


def _sigmoid_sample(logits, tau=1):
    """
    Implementation of Bernouilli reparametrization based on Maddison et al. 2017
    """
    dims = logits.dim()
    logistic_noise = _sample_logistic(logits.size(), out=logits.data.new())
    y = logits + logistic_noise
    return th.sigmoid(y / tau)


def gumbel_sigmoid(logits, ones_tensor, zeros_tensor, tau=1, hard=False):

    shape = logits.size()
    y_soft = _sigmoid_sample(logits, tau=tau)

    if hard:
        y_hard = th.where(y_soft > 0.5, ones_tensor, zeros_tensor)
        y = y_hard.data - y_soft.data + y_soft
    else:
    	y = y_soft
    return y


class ChannelBatchNorm1d(_BatchNorm):
    r"""Applies Batch Normalization over a 2D or 3D input (a mini-batch of 1D
    inputs with optional additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`_ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size).

    By default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momemtum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`L` from input of size :math:`(N, L)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm1d(100, affine=False)
        >>> input = torch.randn(20, 100)
        >>> output = m(input)

    .. _`Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`:
        https://arxiv.org/abs/1502.03167
    """
    def __init__(self, num_channels, num_features, *args, **kwargs):
        super(ChannelBatchNorm1d, self).__init__(num_channels*num_features, *args, **kwargs)
        self.num_channels = num_channels
        self.num_features = num_features

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        _input = input.contiguous().view(-1, self.num_channels * self.num_features)
        output = super(ChannelBatchNorm1d, self).forward(_input)
        return output.view(-1, self.num_channels, self.num_features)


class ParallelBatchNorm1d(th.nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):

        super(ParallelBatchNorm1d, self).__init__()

        # self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = Parameter(th.Tensor(num_features))
            self.bias = Parameter(th.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', th.zeros(1))
            self.register_buffer('running_var', th.ones(1))
            self.register_buffer('num_batches_tracked', th.tensor(0, dtype=th.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            th.nn.init.uniform_(self.weight)
            th.nn.init.zeros_(self.bias)

    def forward(self, input):

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0])
            # use biased var in train
            var = input.var([0], unbiased=False)

            n = input.numel()


            with th.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var


        input = (input - mean) / (th.sqrt(var + self.eps))
        if self.affine:
            input = input * self.weight + self.bias

        return input


def functional_linear3d(input, weight, bias=None):
    r"""
    Apply a linear transformation to the incoming data: :math:`y = xA^T + b`.

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    output = input.transpose(0, 1).matmul(weight)
    if bias is not None:
        output += bias.unsqueeze(1)
    return output.transpose(0, 1)


class Linear3D(th.nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            `(out_features x in_features)`
        bias:   the learnable bias of the module of shape `(out_features)`

    Examples::

        >>> m = nn.Linear(3, 20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, channels, in_features, out_features, batch_size=-1, bias=True, noise=False):
        super(Linear3D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.channels = channels
        if noise:
            self.in_features += 1
        self.weight = Parameter(th.Tensor(channels, self.in_features, out_features))
        if bias:
            self.bias = Parameter(th.Tensor(channels, out_features))
        else:
            self.register_parameter('bias', None)
        if noise:
            self.register_buffer("noise", th.Tensor(batch_size, channels, 1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj_matrix=None, permutation_matrix=None):

        input_ = [input]

        if input.dim() == 2:
            if permutation_matrix is not None:
                input_.append(input.unsqueeze(1).expand([input.shape[0], self.channels, permutation_matrix.shape[1]]))
            elif hasattr(self, "noise"):
                input_.append(input.unsqueeze(1).expand([input.shape[0], self.channels, self.in_features - 1 ]))
            else:
                input_.append(input.unsqueeze(1).expand([input.shape[0], self.channels, self.in_features]))

        if adj_matrix is not None and permutation_matrix is not None:
            input_.append((input_[-1].transpose(0, 1) @ (adj_matrix.t().unsqueeze(2) * permutation_matrix)).transpose(0, 1))
        elif adj_matrix is not None:
            input_.append(input_[-1] * adj_matrix.t().unsqueeze(0))
        elif permutation_matrix is not None:
            input_.append((input_[-1].transpose(0, 1) @ permutation_matrix).t())

        if hasattr(self, 'noise'):
            self.noise.normal_()
            input_.append(th.cat([input_[-1], self.noise], 2))

        return functional_linear3d(input_[-1], self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def apply_filter(self,permutation_matrix):
        transpose_weight = self.weight.transpose(1, 2) @ permutation_matrix
        self.weight = Parameter(transpose_weight.transpose(1, 2))


class GraphSampler(th.nn.Module):
    """Ensemble of all the generators."""

    def __init__(self, n_noises, gnh, graph_size, mask=None):
        """Init the model."""
        super(GraphSampler, self).__init__()

        if not isinstance(graph_size, (list, tuple)):
            self.graph_size = (graph_size, graph_size)
        else:
            self.graph_size = graph_size

        if mask is None:
            mask = 1 - th.eye(*self.graph_size)
        if not (type(mask)==bool and not mask):
            self.register_buffer("mask", mask)

        ones_tensor = th.ones(*self.graph_size)
        self.register_buffer("ones_tensor", ones_tensor)

        zeros_tensor = th.zeros(*self.graph_size)
        self.register_buffer("zeros_tensor", zeros_tensor)


        self.register_buffer("noise_graph_sampler", th.Tensor(1, n_noises))

        layers = []
        layers.append(th.nn.Linear(n_noises, gnh))
        #layers.append(th.nn.BatchNorm1d(gnh))
        layers.append(th.nn.LeakyReLU(.2))
        layers.append(th.nn.Linear(gnh, gnh))
        #layers.append(th.nn.BatchNorm1d(gnh))
        layers.append(th.nn.LeakyReLU(.2))
        layers.append(th.nn.Linear(gnh, gnh))
        #layers.append(th.nn.BatchNorm1d(gnh))
        layers.append(th.nn.LeakyReLU(.2))
        # layers.append(th.nn.Linear(gnh, gnh))
        # layers.append(th.nn.BatchNorm1d(gnh))
        # layers.append(th.nn.LeakyReLU(.2))
        layers.append(th.nn.Linear(gnh, graph_size*graph_size))
        self.layers = th.nn.Sequential(*layers)

        self.reset_parameters()

    def forward(self):

        self.noise_graph_sampler.normal_()

        output_sampler = self.layers(self.noise_graph_sampler).view(*self.graph_size)

        sample_soft = th.sigmoid(output_sampler)
        sample_hard = th.where(output_sampler > 0, self.ones_tensor, self.zeros_tensor)

        #print(output_sampler* self.mask)
        #print(sample_soft* self.mask)
        #print(sample_hard* self.mask)

        sample = sample_hard - sample_soft.data + sample_soft

        return sample * self.mask

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.weight.data.normal_()

class MatrixSampler(th.nn.Module):
    """Matrix Sampler, following a Bernoulli distribution. Differenciable."""
    def __init__(self, graph_size, mask=None, gumble=False):
        super(MatrixSampler, self).__init__()
        if not isinstance(graph_size, (list, tuple)):
            self.graph_size = (graph_size, graph_size)
        else:
            self.graph_size = graph_size
        self.weights = th.nn.Parameter(th.FloatTensor(*self.graph_size))
        self.weights.data.zero_()
        if mask is None:
            mask = 1 - th.eye(*self.graph_size)
        if not (type(mask)==bool and not mask):
            self.register_buffer("mask", mask)
        self.gumble = gumble

        ones_tensor = th.ones(*self.graph_size)
        self.register_buffer("ones_tensor", ones_tensor)

        zeros_tensor = th.zeros(*self.graph_size)
        self.register_buffer("zeros_tensor", zeros_tensor)


    def forward(self, tau=1, drawhard=True):
        """Return a sampled graph."""

        if(self.gumble):

            drawn_proba = gumbel_softmax(th.stack([self.weights.view(-1), -self.weights.view(-1)], 1),
                               tau=tau, hard=drawhard)[:, 0].view(*self.graph_size)
        else:
            drawn_proba = gumbel_sigmoid(2 * self.weights, self.ones_tensor, self.zeros_tensor, tau=tau, hard=drawhard)

        if hasattr(self, "mask"):
            return self.mask * drawn_proba
        else:
            return drawn_proba

    def get_proba(self):
        if hasattr(self, "mask"):
            return th.sigmoid(2 * self.weights) * self.mask
        else:
            return th.sigmoid(2 * self.weights)

    def set_skeleton(self, mask):
        self.register_buffer("mask", mask)


class MatrixSampler2(th.nn.Module):
    """Matrix Sampler, following a Bernoulli distribution. Differenciable."""
    def __init__(self, graph_size, mask=None):
        super(MatrixSampler2, self).__init__()
        if not isinstance(graph_size, (list, tuple)):
            self.graph_size = (graph_size, graph_size)
        else:
            self.graph_size = graph_size
        self.weights = th.nn.Parameter(th.FloatTensor(*self.graph_size))
        self.weights.data.zero_()
        self.v_weights = th.nn.Parameter(th.where(th.eye(*self.graph_size)>0, th.ones(*self.graph_size).fill_(1), th.zeros(*self.graph_size))
                                         .repeat(self.graph_size[1], 1, 1)
                                         .transpose(0, 2))
        if mask is None:
            mask = 1 - th.eye(*self.graph_size)
        if not (type(mask)==bool and not mask):
            self.register_buffer("mask", mask)
    def forward(self, tau=1, drawhard=True):
        """Return a sampled graph."""
        # drawn_proba = gumbel_softmax(th.stack([self.weights.view(-1), -self.weights.view(-1)], 1),
        #                        tau=tau, hard=drawhard)[:, 0].view(*self.graph_size)
        # corr_weights = (drawn_proba.unsqueeze(0) *
        #                 (self.v_weights/ (.5 * self.v_weights.abs().sum(1, keepdim=True)))).sum(0)
        corr_weights = (self.weights.unsqueeze(1) *
                        (self.v_weights/ self.v_weights.abs().sum(1, keepdim=True))).sum(0)
        out_proba = gumbel_softmax(th.stack([corr_weights.view(-1), -corr_weights.view(-1)], 1),
                               tau=tau, hard=drawhard)[:, 0].view(*self.graph_size)
        if hasattr(self, "mask"):
            return self.mask * out_proba
        else:
            return out_proba

    def get_proba(self):
        if hasattr(self, "mask"):
            return th.sigmoid(2 * self.weights) * self.mask
        else:
            return th.sigmoid(2 * self.weights)

    def set_skeleton(self, mask):
        self.register_buffer("mask", mask)


class MatrixSampler3(th.nn.Module):
    """Matrix Sampler, following a Bernoulli distribution. Differenciable."""
    def __init__(self, graph_size, mask=None, gumbel=True, k=None):
        super(MatrixSampler3, self).__init__()
        if not isinstance(graph_size, (list, tuple)):
            self.graph_size = (graph_size, graph_size)
        else:
            self.graph_size = graph_size
        self.k = k if k is not None else self.graph_size[0] - 1
        self.in_weights = th.nn.Parameter(th.FloatTensor(self.graph_size[0], self.k))
        self.out_weights = th.nn.Parameter(th.FloatTensor(self.k, self.graph_size[1]))
        self.in_weights.data.normal_()
        self.out_weights.data.normal_()
        self.gumbel_softmax = gumbel
        if not gumbel:
            ones_tensor = th.ones(*self.graph_size)
            zeros_tensor = th.zeros(*self.graph_size)
            self.register_buffer("ones_tensor", ones_tensor)
            self.register_buffer("zeros_tensor", zeros_tensor)

        if mask is None:
            mask = 1 - th.eye(*self.graph_size)
        if not (type(mask)==bool and not mask):
            self.register_buffer("mask", mask)

    def forward(self, tau=1, drawhard=True):
        """Return a sampled graph."""
        corr_weights = self.in_weights @ self.out_weights
        if self.gumbel_softmax:
            out_sample = gumbel_softmax(th.stack([corr_weights.view(-1), -corr_weights.view(-1)], 1),
                                       tau=tau, hard=drawhard)[:, 0].view(*self.graph_size)
        else:
            sample_soft = th.sigmoid(corr_weights)
            sample_hard = th.where(corr_weights > 0,
                                   self.ones_tensor, self.zeros_tensor)
            out_sample = sample_hard - sample_soft.data + sample_soft

        if hasattr(self, "mask"):
            return self.mask * out_sample
        else:
            return out_sample

    def get_proba(self):
        if hasattr(self, "mask"):
            return th.sigmoid(2 * (self.in_weights @ self.out_weights)) * self.mask
        else:
            return th.sigmoid(2 * (self.in_weights @ self.out_weights))

    def set_skeleton(self, mask):
        self.register_buffer("mask", mask)


class SimpleMatrixConnection(th.nn.Module):
    """Matrix Sampler, following a Bernoulli distribution. Differenciable."""

    def __init__(self, graph_size, mask=None):
        super(SimpleMatrixConnection, self).__init__()
        if not isinstance(graph_size, (list, tuple)):
            self.graph_size = (graph_size, graph_size)
        else:
            self.graph_size = graph_size
        self.weights = th.nn.Parameter(th.FloatTensor(*self.graph_size))
        self.weights.data.normal_()

        if mask is None:
            mask = 1 - th.eye(*self.graph_size)
        if not (type(mask) == bool and not mask):
            self.register_buffer("mask", mask)

        ones_tensor = th.ones(*self.graph_size)
        self.register_buffer("ones_tensor", ones_tensor)

        zeros_tensor = th.zeros(*self.graph_size)
        self.register_buffer("zeros_tensor", zeros_tensor)

    def forward(self):
        """Return a sampled graph."""

        sample_soft = th.sigmoid(2 * self.weights)

        sample_hard = th.where(self.weights > 0, self.ones_tensor, self.zeros_tensor)
        sample = sample_hard - sample_soft.data + sample_soft

        if hasattr(self, "mask"):
            return self.mask * sample_soft
        else:
            return sample_soft

    def get_proba(self):
        if hasattr(self, "mask"):
            return th.sigmoid(2 * self.weights) * self.mask
        else:
            return th.sigmoid(2 * self.weights)


