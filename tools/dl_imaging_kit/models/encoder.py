from typing import *

import torch
from torch import nn
import torchvision

from dl_imaging_kit.models import blocks


class BaseEncoder(nn.Module):

    def __init__(self,
                 conv_blocks: List[blocks.ConvBlock],
                 stem: Optional[blocks.ConvBlock] = None,
                 add_pool_to_blocks: bool = True):
        super(BaseEncoder, self).__init__()

        self.stem = stem
        if add_pool_to_blocks:
            downsampling_blocks = [blocks.DownSamplingBlock(block) for block in conv_blocks]
        else:
            downsampling_blocks = conv_blocks

        self.blocks = nn.Sequential(*downsampling_blocks)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_intermediate = dict()
        if self.stem is not None:
            x = self.stem(x)
            x_intermediate['stem'] = x

        for i, block in enumerate(self.blocks, start=1):
            x = block(x)
            x_intermediate[f'block_{i}'] = x

        return x, x_intermediate


class ResnetEncoder(BaseEncoder):

    def __init__(self,
                 resnet_class: Callable = torchvision.models.resnet101,
                 pretrained=True,
                 **kwargs):
        resnet = resnet_class(pretrained=pretrained, **kwargs)
        stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        blocks = [resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]
        super(ResnetEncoder, self).__init__(blocks, stem=stem, add_pool_to_blocks=False)


class StandardCNNEncoder(BaseEncoder):

    def __init__(self,
                 in_channels: int = 3,
                 channels_per_depth: Sequence[int] = (64, 128, 256, 512, 1024),
                 convs_per_depth: Sequence[int] = (2, 2, 2, 2, 2),
                 block_class: type = blocks.ConvBlock,
                 stem: Optional[blocks.ConvBlock] = None,
                 kernel_size: int = 3,
                 block_kwargs: Optional[dict] = None):
        assert len(channels_per_depth) == len(convs_per_depth), \
            'channels_per_depth and convs_per_depth must be equal length'

        first_block = block_class(in_channels,
                                  channels_per_depth[0],
                                  kernel_size=kernel_size,
                                  n_convs=convs_per_depth[0],
                                  **(block_kwargs or dict()))
        conv_blocks = [first_block]

        for i, (n_channels, n_convs) in enumerate(zip(channels_per_depth[1:], convs_per_depth[1:]), start=1):
            previous_n_channels = channels_per_depth[i - 1]
            conv_block = block_class(previous_n_channels,
                                     n_channels,
                                     kernel_size=kernel_size,
                                     n_convs=convs_per_depth[i],
                                     **(block_kwargs or dict()))
            conv_blocks.append(conv_block)

        super(StandardCNNEncoder, self).__init__(conv_blocks, stem)


class VovNetEncoder(StandardCNNEncoder):

    def __init__(self,
                 in_channels: int,
                 channels_per_depth: Sequence[int] = (32, 64, 128, 256),
                 convs_per_depth: Sequence[int] = (5, 5, 5, 5),
                 stem: Optional[blocks.ConvBlock] = None,
                 squeeze_and_excitation_block: blocks.SqueezeAndExcitation = blocks.EfficientSqueezeAndExcitation):
        super(VovNetEncoder, self).__init__(in_channels,
                                            channels_per_depth,
                                            convs_per_depth,
                                            block_class=blocks.OSABlock,
                                            stem=stem,
                                            block_kwargs={'squeeze_and_excitation_block': squeeze_and_excitation_block}
                                            )
