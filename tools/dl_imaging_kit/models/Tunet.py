from typing import *

import torch
import sys
from torch import nn
sys.path.append("/home/edfo/local_libs/tools/")
from dl_imaging_kit.models import blocks


class TUNet(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 n_transformer: int = 12,
                 n_at_heads: int = 12,
                 hidden_size: int = 768,
                 at_dropout=0.0,
                 mlp_dim: int = 3072,
                 mlp_dropout=0.1,
                 output_activation: Optional[Callable] = None,
                 channels_per_depth: Sequence[int] = (64, 128, 256, 512, 1024),
                 convs_per_depth: Sequence[int] = (2, 2, 2, 2, 2),
                 block_class: type = blocks.ConvBlock,
                 stem: Optional[blocks.ConvBlock] = None,
                 kernel_size: int = 3,
                 block_kwargs: Optional[dict] = None):
        assert len(channels_per_depth) == len(convs_per_depth)
        assert issubclass(block_class, blocks.ConvBlock), 'param block_class should inherit ConvBlock'
        super(TUNet, self).__init__()

        self.stem = stem

        first_block = block_class(in_channels,
                                  channels_per_depth[0],
                                  kernel_size=kernel_size,
                                  n_convs=convs_per_depth[0],
                                  **(block_kwargs or dict()))
        last_block = block_class(channels_per_depth[0] * 2,
                                 channels_per_depth[0],
                                 kernel_size=kernel_size,
                                 n_convs=convs_per_depth[0],
                                 **(block_kwargs or dict()))

        encoder_convs = [blocks.DownSamplingBlock(first_block)]
        decoder_convs = [blocks.UpSamplingBlock(last_block)]
        transblocks = [blocks.Embeddings(patch_size=16, in_channels=1)]
        for i in range(0, n_transformer):
            transblocks.append(blocks.TransBlock())
        transblocks.append(blocks.Dislodging(out_channels=channels_per_depth[-1]))

        for i, (n_channels, n_convs) in enumerate(zip(channels_per_depth[1:], convs_per_depth[1:]), start=1):
            previous_n_channels = channels_per_depth[i - 1]
            encoder_conv = block_class(previous_n_channels,
                                       n_channels,
                                       kernel_size=kernel_size,
                                       n_convs=convs_per_depth[i],
                                       **(block_kwargs or dict()))
            encoder_convs.append(blocks.DownSamplingBlock(encoder_conv))

            decoder_conv = block_class(n_channels * (2 if i < (len(channels_per_depth) - 1) else 1),
                                       previous_n_channels,
                                       kernel_size=kernel_size,
                                       n_convs=convs_per_depth[i],
                                       **(block_kwargs or dict()))
            decoder_convs.insert(0, blocks.UpSamplingBlock(decoder_conv))

        self.transformer = nn.Sequential(*transblocks)
        self.encoder = nn.Sequential(*encoder_convs)
        self.decoder = nn.Sequential(*decoder_convs)
        self.last_conv = nn.Conv2d(channels_per_depth[0], out_channels, kernel_size=1)
        self.output_activation = output_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x
        if self.stem is not None:
            x = self.stem(x)

        x_intermediate = list()
        for block in self.encoder[:]:
            x = block(x)
            x_intermediate.append(x)

        for i, block in enumerate(self.transformer):
            if i == 0:
                xt = block(xt)
            elif i == len(self.transformer) - 1:
                xt = block(xt)
            else:
                xt, _ = block(xt)
        x = xt

        for i, block in enumerate(self.decoder):
            if i > 0:
                x = torch.cat([x_intermediate[-(i + 1)], x], dim=1)
            x = block(x)

        x = self.last_conv(x)
        if self.output_activation is not None:
            x = self.output_activation(x)

        return x


if __name__ == '__main__':
    model = TUNet(in_channels=1, out_channels=4,
                 convs_per_depth=(5, 5, 5, 5),
                 channels_per_depth=(32, 64, 128, 256),
                 block_class=blocks.OSABlock,
                 block_kwargs={'squeeze_and_excitation_block': blocks.EfficientSqueezeAndExcitation})
    x = torch.randn(4, 1, 256, 256)
    print(model(x).shape)
