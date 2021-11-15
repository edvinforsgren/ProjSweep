import math
from typing import *

import numpy as np
import torch
from torch import nn
import sys
sys.path.append("/home/edfo/local_libs/tools/")
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from dl_imaging_kit.models.activations import HardSigmoid, RadixSoftMax


class SamePadConv2D(nn.Conv2d):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 pad_mode: str = 'reflect',
                 pad_value: float = 0,
                 **kwargs):  # Catch kwargs to avoid unexpected key-word errors.
        super(SamePadConv2D, self).__init__(in_channels,
                                            out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            dilation=dilation,
                                            groups=groups,
                                            bias=bias)
        self.pad_mode = pad_mode
        self.pad_value = pad_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndimension() == 4, 'Expected 4-dimensional tensor'
        _, _, h, w = x.shape
        pad_x = _get_padding(w, self.kernel_size[1], self.stride[1], self.dilation[1])
        pad_y = _get_padding(h, self.kernel_size[0], self.stride[0], self.dilation[0])
        x = F.pad(x, (pad_x, pad_x, pad_y, pad_y), self.pad_mode, self.pad_value)
        x = super(SamePadConv2D, self).forward(x)
        assert x.shape[-2:] == x.shape[-2:], 'Shape mismatch'
        return x


class ConvNormalizeActivate(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 activation: type = nn.ReLU,
                 pre_activate: bool = False,
                 padding_mode: str = 'same',
                 normalization_constructor: type = nn.BatchNorm2d,
                 **kwargs):
        super(ConvNormalizeActivate, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pre_activate = pre_activate
        self.padding_mode = padding_mode
        if padding_mode == 'same':
            conv = SamePadConv2D(in_channels, out_channels, kernel_size, **kwargs)
        else:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        normalizer = normalization_constructor(out_channels)

        if pre_activate:
            self.conv_normalize_activate = nn.Sequential(activation(), conv, normalizer)
        else:
            self.conv_normalize_activate = nn.Sequential(conv, normalizer, activation())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_normalize_activate(x)

class DepthwiseSeperableConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 activation: type = nn.ReLU,
                 pre_activate: bool = False,
                 padding: int = 1):
        super().__init__()

        #Depthwise conv
        depthwise_conv = ConvNormalizeActivate(in_channels, in_channels, kernel_size, activation=activation, pre_activate=pre_activate, stride=stride, padding_mode="", padding=padding, bias=False, groups=in_channels)

        #Pointwise conv
        pointwise_conv = ConvNormalizeActivate(in_channels, out_channels, kernel_size=1, stride=1, bias=False, padding_mode="", padding=0)

        self.depthwise_sep_conv= nn.Sequential(depthwise_conv, pointwise_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.depthwise_sep_conv(x)


def basic_conv(in_planes, out_planes, kernel_size, stride=1, padding=1, groups=1,
               with_bn=True, with_relu=True):
    """convolution with bn and relu"""
    module = []
    has_bias = not with_bn
    module.append(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                  bias=has_bias)
    )
    if with_bn:
        module.append(nn.BatchNorm2d(out_planes))
    if with_relu:
        module.append(nn.ReLU())
    return nn.Sequential(*module)

def depthwise_separable_conv(in_planes, out_planes, kernel_size, stride=1, padding=1, groups=1,
                             with_bn=True, with_relu=True):
    """depthwise separable convolution with bn and relu"""
    del groups

    module = []
    module.extend([
        basic_conv(in_planes, in_planes, kernel_size, stride, padding, groups=in_planes,
                   with_bn=True, with_relu=True),
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
    ])
    if with_bn:
        module.append(nn.BatchNorm2d(out_planes))
    if with_relu:
        module.append(nn.ReLU())
    return nn.Sequential(*module)


class SqueezeAndExcitation(nn.Module):

    def __init__(self,
                 channels: int,
                 reduction_factor: float,
                 activation: type = nn.ReLU,
                 use_hard_sigmoid: bool = False):
        super(SqueezeAndExcitation, self).__init__()
        self.channels = channels
        self.reduction_factor = reduction_factor
        self.use_hard_sigmoid = use_hard_sigmoid
        reduced_channels = int(channels / reduction_factor)
        self.reduce = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, kernel_size=1, padding=0),
            activation(),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, padding=0),
            HardSigmoid() if use_hard_sigmoid else nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_reduced = self.reduce(x)
        x = x * x_reduced
        return x


class EfficientSqueezeAndExcitation(SqueezeAndExcitation):

    def __init__(self, channels: int, use_hard_sigmoid: bool = True):
        super(EfficientSqueezeAndExcitation, self).__init__(channels, reduction_factor=1)
        self.reduce = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0),
            HardSigmoid() if use_hard_sigmoid else nn.Sigmoid()
        )


class Attention(nn.Module):

    def __init__(self,
                 n_at_heads: int = 12,
                 hidden_size: int = 768,
                 at_dropout=0.0,
                 vis=None):
        super(Attention, self).__init__()
        self.vis = vis
        self.n_at_heads = n_at_heads
        self.hidden_size = hidden_size
        self.at_head_size = int(hidden_size / n_at_heads)
        self.all_head_size = self.n_at_heads * self.at_head_size

        self.q = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.val = nn.Linear(self.hidden_size, self.all_head_size)

        self.out = nn.Linear(self.hidden_size, self.all_head_size)
        self.at_dropout = nn.Dropout(at_dropout)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x = x.size()[:-1] + (self.n_at_heads, self.at_head_size)
        x = x.view(*new_x)
        return x.permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mq_layer = self.q(x)
        mk_layer = self.key(x)
        mv_layer = self.val(x)

        q_layer = self.transpose_for_scores(mq_layer)
        k_layer = self.transpose_for_scores(mk_layer)
        v_layer = self.transpose_for_scores(mv_layer)

        att_scores = (torch.matmul(q_layer, k_layer.transpose(-1, -2))) / math.sqrt(self.at_head_size)
        att_probs = self.softmax(att_scores)
        weights = att_probs if self.vis else None
        att_probs = self.at_dropout(att_probs)

        cont_layer = torch.matmul(att_probs, v_layer)
        cont_layer = cont_layer.permute(0, 2, 1, 3).contiguous()
        new_cont_layer_shape = cont_layer.size()[:-2] + (self.all_head_size, )
        cont_layer = cont_layer.view(*new_cont_layer_shape)
        att_out = self.out(cont_layer)
        att_out = self.at_dropout(att_out)
        return att_out, weights


class mlp(nn.Module):
    def __init__(self,
                 hidden_size: int = 768,
                 mlp_dim: int = 3072,
                 mlp_dropout=0.1,
                 activation=F.gelu):
        super(mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.actf = activation
        self.dropout = nn.Dropout(mlp_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.actf(x)
        x = self.fc2(x)
        x = self.actf(x)
        x = self.dropout(x)
        return x


class TransBlock(nn.Module):
    def __init__(self,
                 hidden_size: int = 768,
                 attention_norm=nn.LayerNorm,
                 mlp_norm=nn.LayerNorm,
                 eps=1e-6
                 ):
        super(TransBlock, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.attention_norm = attention_norm(self.hidden_size, self.eps)
        self.mlp_norm = mlp_norm(self.hidden_size, self.eps)
        self.mlp = mlp()
        self.attn = Attention()

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x+h

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x+h
        return x, weights


class Embeddings(nn.Module):
    def __init__(self,
                 #img_size=256,
                 in_channels: int = 1,
                 patch_size=16,
                 hidden_size=768,
                 dropout_rate=0.1):
        super(Embeddings, self).__init__()
        #self.img_size = _pair(img_size)
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_size = _pair(patch_size)
        #n_patches = (self.img_size[0] // patch_size[0]) * (self.img_size[1] // patch_size[1])
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                          out_channels=self.hidden_size,
                                          kernel_size=patch_size,
                                          stride=patch_size)
        #self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, self.hidden_size))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        img_size = x.shape[2:]
        n_patches = (img_size[0] // self.patch_size[0] * img_size[1] // self.patch_size[1])
        patch_embeddings = nn.Conv2d(in_channels=self.in_channels,
                                          out_channels=self.hidden_size,
                                          kernel_size=self.patch_size,
                                          stride=self.patch_size)
        patch_embeddings = patch_embeddings.cuda()
        x = patch_embeddings(x)
        position_embeddings = nn.Parameter(torch.zeros(1, n_patches, self.hidden_size))
        position_embeddings = position_embeddings.cuda()
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Dislodging(nn.Module):
    def __init__(self,
                 hidden_size: int = 768,
                 out_channels: int = 256):
        super(Dislodging, self).__init__()
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.conv = ConvNormalizeActivate(self.hidden_size,
                                          self.out_channels,
                                          kernel_size=3,
                                          padding_mode=False,
                                          padding=1)


    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv(x)
        return x


# class DecoderCUP(nn.Module):
#     def __init__(self,
#                  hidden_size: int = 768,
#                  head_channels: int = 512,
#                  decoder_channels=(256, 128, 64, 32, 16),
#                  n_skip=0,
#                  skip_channels=None):
#         super(DecoderCUP, self).__init__()
#         self.hidden_size = hidden_size
#         self.head_channels = head_channels
#         self.conv = ConvNormalizeActivate(self.hidden_size,
#                                           self.head_channels,
#                                           kernel_size=3,
#                                           padding_mode=False,
#                                           padding=1)
#         in_channels = [self.head_channels] + list(decoder_channels[:-1])
#         out_channels = decoder_channels
#
#         if n_skip != 0:
#             for i in range(4-n_skip):
#                 skip_channels[3-i] = 0
#         else:
#             skip_channels = [0, 0, 0, 0]
#
#         blocks = []

class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 n_convs: int,
                 first_conv_channels: int = 1,
                 **kwargs):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_convs = n_convs
        self.first_conv_channels = first_conv_channels
        self.first_conv = ConvNormalizeActivate(in_channels, out_channels, first_conv_channels, **kwargs)
        convs = list()
        for _ in range(n_convs - 1):
            convs.append(ConvNormalizeActivate(out_channels, out_channels, kernel_size, **kwargs))

        self.convs = nn.Sequential(*convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(self.first_conv(x))


class ResBlock(ConvBlock):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 n_convs: int,
                 skip_connection_weight: float = 1,
                 **kwargs):
        super(ResBlock, self).__init__(in_channels, out_channels, kernel_size, n_convs, **kwargs)
        self.skip_connection_weight = skip_connection_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x = self.first_conv(x)
        x = self.convs(x)
        return x + x_in * self.skip_connection_weight


class OSABlock(ConvBlock):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 n_convs: int,
                 use_identity_mapping: bool = True,
                 squeeze_and_excitation_block: Optional[type] = None,
                 **kwargs):
        super(OSABlock, self).__init__(in_channels, out_channels, kernel_size, n_convs, **kwargs)
        self.conv1x1 = SamePadConv2D(out_channels * (n_convs - 1), out_channels, kernel_size=1, **kwargs)
        self.use_identity_mapping = use_identity_mapping

        if squeeze_and_excitation_block is not None:
            assert issubclass(squeeze_and_excitation_block, SqueezeAndExcitation), \
                'param squeeze_and_excitation_block must inherit SqueezeAndExcitation'
            self.se_block = squeeze_and_excitation_block(out_channels)
        else:
            self.se_block = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x = self.first_conv(x)
        x_intermediate = list()
        for conv in self.convs:
            x = conv(x)
            x_intermediate.append(x)

        x = torch.cat(x_intermediate, dim=1)
        x = self.conv1x1(x)

        if self.se_block is not None:
            x = self.se_block(x)

        if self.use_identity_mapping:
            x = x + x_in

        return x


class SplitAttentionConv2D(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 cardinality: int = 1,
                 radix: int = 2,
                 reduction_factor: int = 4,
                 max_inter_channels: Optional[int] = 32,
                 base_convblock_class: type = ConvNormalizeActivate,
                 normalization_constructor: Optional[type] = nn.BatchNorm2d,
                 activation: Callable = nn.ReLU,
                 use_identity_mapping: bool = True,
                 **kwargs):
        super(SplitAttentionConv2D, self).__init__()
        assert radix > 1, 'param radix needs to be larger than 1 in order to use split-attention'
        if use_identity_mapping:
            assert in_channels == out_channels, 'params in_channels and out_channels must be equal if ' \
                                                'param use_identity_mapping is true.'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.radix = radix
        self.cardinality = cardinality
        self.reduction_factor = reduction_factor
        self.max_inter_channels = max_inter_channels
        self.use_identity_mapping = use_identity_mapping

        inter_channels = in_channels * radix // reduction_factor
        if max_inter_channels is not None:
            inter_channels = max(inter_channels, max_inter_channels)
        self.conv_block = base_convblock_class(in_channels,
                                               out_channels * radix,
                                               kernel_size,
                                               groups=cardinality * radix,
                                               normalization_constructor=normalization_constructor,
                                               activation=activation,
                                               **kwargs)

        if isinstance(activation, type):
            self.activation = activation()
            assert callable(self.activation), 'Activation instance is not callable.'
        elif callable(activation):
            self.activation = activation
        else:
            raise ValueError('param activation needs to be class or callable')
        self.fc1 = nn.Conv2d(out_channels, inter_channels, kernel_size=1, groups=cardinality)
        self.fc2 = nn.Conv2d(inter_channels, out_channels * radix, kernel_size=1, groups=cardinality)
        self.radix_softmax = RadixSoftMax(radix, cardinality)

        if normalization_constructor is not None:
            self.normalization = normalization_constructor(inter_channels)
        else:
            self.normalization = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        x = self.conv_block(x)
        n_batch, n_channels_after_conv, _, _ = x.shape

        x_splits = torch.split(x, n_channels_after_conv // self.radix, dim=1)  # R x [N x (C / R) x 1 x 1]
        x = sum(x_splits)  # N x (C / R) x H x W
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))  # N x (C / R) x 1 x 1
        x = self.fc1(x)
        if self.normalization is not None:
            x = self.normalization(x)

        x = self.activation(x)
        x = self.fc2(x)
        x = self.radix_softmax(x).view(n_batch, -1, 1, 1)  # N x (C / R) x 1 x 1
        attention_splits = torch.split(x, n_channels_after_conv // self.radix, dim=1)
        x = sum(attention_split * x_split for (attention_split, x_split) in zip(attention_splits, x_splits))
        if self.use_identity_mapping:
            x_in + x
        return x.contiguous()


class DownSamplingBlock(nn.Module):

    def __init__(self,
                 conv_block: ConvBlock,
                 pooling_size: int = 2,
                 stride: int = 2):
        super(DownSamplingBlock, self).__init__()
        self.conv_block = conv_block
        self.pool = nn.MaxPool2d(pooling_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv_block(x))


class UpSamplingBlock(nn.Module):

    def __init__(self,
                 conv_block: ConvBlock,
                 scale_factor: int = 2):
        super(UpSamplingBlock, self).__init__()
        self.conv_block = conv_block
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(self.conv_block(x))


def _get_padding(size: int, kernel_size: int, stride: int, dilation: int) -> int:
    padding = (dilation * (kernel_size - 1) + (stride - 1) * (size - 1)) // 2
    return padding


"""
Implementation of the Atrous Spatial Pyramid Pooling (ASPP) from Panoptic deeplab paper
"""


class ASPPConv(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dilation
                 ):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super().__init__(*modules)


class ASPPooling(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int
                 ):
        super().__init__()

        self.aspp_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU()
        )

    def set_image_pooling(self,
                          pool_size):
        self.aspp_pooling[0] = nn.AvgPool2d(kernel_size=pool_size, stride=1)


    def forward(self, x):
        size = x.shape[-2:]
        x = self.aspp_pooling(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)


class ASPP(nn.Module):
    def __init__(self,
                 in_channels: int = 64,
                 out_channels: int = 256,
                 atrous_rates: Sequence = (5, 6, 9)
                 ):
        super().__init__()

        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.projects = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def set_image_pooling(self, pool_size):
        self.convs[-1].set_image_pooling(pool_size)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.projects(res)



if __name__ == '__main__':
    x = torch.randn(2, 3, 768, 768)

    # block = OSABlock(3, 4, 3, 5,
    #                  squeeze_and_excitation_block=EfficientSqueezeAndExcitation(4),
    #                  activation=nn.Sigmoid)
    block = Attention()
    block(x)
