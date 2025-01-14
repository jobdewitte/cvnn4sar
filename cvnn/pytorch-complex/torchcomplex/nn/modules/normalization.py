import torch
from torch.nn.parameter import Parameter
from torch.nn import ParameterList
from torch.nn.modules.module import Module

from .. import functional as cF
from torch.nn import init

from torch import Tensor, Size
from typing import Union, List, Tuple

__all__ = ['GroupNorm']
class GroupNorm(Module):
    r"""Applies Group Normalization over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Group Normalization <https://arxiv.org/abs/1803.08494>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The input channels are separated into :attr:`num_groups` groups, each containing
    ``num_channels / num_groups`` channels. :attr:`num_channels` must be divisible by
    :attr:`num_groups`. The mean and standard-deviation are calculated
    separately over the each group. :math:`\gamma` and :math:`\beta` are learnable
    per-channel affine transform parameter vectors of size :attr:`num_channels` if
    :attr:`affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        num_groups (int): number of groups to separate the channels into
        num_channels (int): number of channels expected in input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to ``True``, this module
            has learnable per-channel affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, C, *)` where :math:`C=\text{num\_channels}`
        - Output: :math:`(N, C, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 6, 10, 10)
        >>> # Separate 6 channels into 3 groups
        >>> m = nn.GroupNorm(3, 6)
        >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
        >>> m = nn.GroupNorm(6, 6)
        >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
        >>> m = nn.GroupNorm(1, 6)
        >>> # Activating the module
        >>> output = m(input)
    """

    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine']
    num_groups: int
    num_channels: int
    eps: float
    affine: bool

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True,
                 device=None, dtype=None, naive: bool = True, complex_weights: bool = True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.naive = naive
        self.complex_weights = complex_weights
        
        if naive:
            if self.affine:
                if complex_weights:
                    self.weight = Parameter(torch.Tensor(num_channels).to(torch.cfloat))
                    self.bias = Parameter(torch.Tensor(num_channels).to(torch.cfloat))
                else:
                    self.weight = ParameterList([Parameter(torch.Tensor(num_channels)), Parameter(torch.Tensor(num_channels))])
                    self.bias = ParameterList([Parameter(torch.Tensor(num_channels)), Parameter(torch.Tensor(num_channels))])
            else:
                self.register_parameter('weight', None)
                self.register_parameter('bias', None)
        else:
            raise NotImplementedError

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.naive:
            if self.affine:
                if self.complex_weights:
                    init.ones_(self.weight)
                    init.zeros_(self.bias)
                else:
                    init.ones_(self.weight[0])
                    init.zeros_(self.bias[0])
                    init.ones_(self.weight[1])
                    init.zeros_(self.bias[1])
        else:
            raise NotImplementedError

    def forward(self, input: Tensor) -> Tensor:
        return cF.group_norm(
            input, self.num_groups, self.weight, self.bias, self.eps, self.naive)

    def extra_repr(self) -> str:
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)