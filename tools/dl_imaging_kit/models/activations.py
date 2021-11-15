import torch
from torch import nn
from torch.nn import functional as F


class Mish(nn.Module):

    """ See :func:`~dl_imaging_kit.models.blocks.mish` """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mish(x)


class HardSigmoid(nn.Module):

    """ See :func:`~dl_imaging_kit.models.blocks.hard_sigmoid` """

    def __init__(self, inplace: bool = True):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return hard_sigmoid(x, self.inplace)


class Swish(nn.Module):

    """ See :func:`~dl_imaging_kit.models.blocks.swish` """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swish(x)


class RadixSoftMax(nn.Module):

    """ See :func:`~dl_imaging_kit.models.blocks.radix_softmax` """

    def __init__(self, radix: int, cardinality: int):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = radix_softmax(x, self.cardinality, self.radix)
        return x


def mish(x: torch.Tensor) -> torch.Tensor:
    """ Mish-activation according to Misra 2019

    Parameters
    ----------
    x : torch.Tensor
        Input-tensor.

    References
    ----------
    Misra, Diganta. "Mish: A self regularized non-monotonic neural activation function."
    arXiv preprint arXiv:1908.08681 (2019).

    Returns
    -------
    torch.Tensor
        Same shape as input tensor.
    """
    return x * (torch.tanh(F.softplus(x)))


def swish(x: torch.Tensor) -> torch.Tensor:
    """ Swish/SiLU-activation.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    References
    ----------
    Ramachandran, Prajit, Barret Zoph, and Quoc V. Le. "Searching for activation functions."
    arXiv preprint arXiv:1710.05941 (2017).
    Hendrycks, Dan, and Kevin Gimpel. "Bridging nonlinearities and stochastic regularizers
    with gaussian error linear units." (2016).

    Returns
    -------
    torch.Tensor
        Same shape as input tensor.
    """

    return x * F.sigmoid(x)


def hard_sigmoid(x: torch.Tensor, inplace: bool = True) -> torch.Tensor:
    """ Piece-wise linear approximation of sigmoid.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    inplace : bool
        If True, perform relu in-place.

    Returns
    -------
    torch.Tensor
        Same shape as output.
    """
    return F.relu6(x + 3.0, inplace=inplace) / 6.0


def radix_softmax(x: torch.Tensor,
                  cardinality: int,
                  radix: int) -> torch.Tensor:
    """ Radix-major softmax as used in ResNest (see Zhang et al.)

    Parameters
    ----------
    x : torch.Tensor
        Input N x C x H x W-tensor
    cardinality : int
        Number of cardinal groups.
    radix : int
        Number of splits of each cardinal group.

    References
    ----------
    Zhang, Hang, et al. "Resnest: Split-attention networks." arXiv preprint arXiv:2004.08955 (2020).

    Returns
    -------
    torch.Tensor
    """
    batch = x.size(0)
    if radix > 1:
        x = x.view(batch, cardinality, radix, -1).transpose(1, 2)
        x = F.softmax(x, dim=1)
        x = x.reshape(batch, -1)
    else:
        x = torch.sigmoid(x)
    return x
