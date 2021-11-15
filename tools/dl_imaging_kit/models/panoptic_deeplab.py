import torch
from torch import nn
from torch.nn import functional as F
from dl_imaging_kit.models.blocks import DepthwiseSeperableConv as DSConv
from dl_imaging_kit.models.blocks import ASPP
from collections import OrderedDict

class PanopticDeepLab(nn.Module):
    """
    Implementation of the deeplab panoptics arcitecture.
    For reference: https://arxiv.org/abs/1911.10194
    """

    def __init__(self,
                 fpn_feature_key,
                 fpn_feature_channels,
                 fpn_low_level_feature_keys,
                 backbone: nn.Module,
                 sem_dec_feature_cls=[64, 32],
                 ins_dec_feature_cls=[32, 16],
                 low_level_channels=(512, 256),
                 sem_dec_channels=256,
                 ins_dec_channels=128
                 ):
        super().__init__()

        self.backbone = backbone

        self.semantic_decoder = DeepLabEncoder(fpn_feature_key, fpn_feature_channels, fpn_low_level_feature_keys,
                                               low_level_channels=low_level_channels,
                                               low_level_project_channels=sem_dec_feature_cls,
                                               decoder_channels=sem_dec_channels)
        self.instance_decoder = DeepLabEncoder(fpn_feature_key, fpn_feature_channels, fpn_low_level_feature_keys,
                                               low_level_channels=low_level_channels,
                                               low_level_project_channels=ins_dec_feature_cls,
                                               decoder_channels=ins_dec_channels)


        self.semantic_predictor = DeepLabPredHead(in_channels=256, internal_channels=256, out_channels=3,
                                                  key_name="semantic")
        self.instance_predictor = {
            "center": DeepLabPredHead(in_channels=128, internal_channels=32, out_channels=1,
                                      key_name="center"),
            "regression": DeepLabPredHead(in_channels=128, internal_channels=32, out_channels=2,
                                          key_name="offset")
        }
        self.instance_predictor = torch.nn.ModuleDict(self.instance_predictor)


        self.post_processing = None

    def _upsample_predictions(self, pred, input_shape):
        """Upsamples final prediction.
        Args:
            pred (dict): stores all output of the segmentation model.
            input_shape (tuple): spatial resolution of the desired shape.
        Returns:
            result (OrderedDict): upsampled dictionary.
        """
        result = OrderedDict()
        for key in pred.keys():
            out = F.interpolate(pred[key], size=input_shape, mode='bilinear', align_corners=True)

            if 'offset' in key:
                scale = (input_shape[0] - 1) // (pred[key].shape[2] - 1)
                out *= scale
            result[key] = out
        return result

    def forward(self, x):
        input_size = x[0].shape[-2:]

        output = self.backbone(x)

        semantic_out = self.semantic_decoder(output[1])
        semantic_out = self.semantic_predictor(semantic_out)

        instance_out = self.instance_decoder(output[1])
        center_pred = self.instance_predictor['center'](instance_out)
        center_reg = self.instance_predictor['regression'](instance_out)

        output_dict = dict(semantic_out)
        output_dict.update(center_pred)
        output_dict.update(center_reg)

        output_dict = self._upsample_predictions(output_dict, input_size)

        return output_dict


class DeepLabEncoder(nn.Module):

    def __init__(self,
                 fpn_feature_key,
                 fpn_feature_channels,
                 fpn_low_level_feature_keys,
                 low_level_channels,
                 low_level_project_channels,
                 decoder_channels=256,
                 aspp_channels = None
                 ):
        super().__init__()

        self.fpn_feature_key = fpn_feature_key
        self.fpn_low_level_feature_keys = fpn_low_level_feature_keys
        self.decoder_stages = len(low_level_channels)
        if aspp_channels is None:
            aspp_channels = decoder_channels

        self.aspp = ASPP(in_channels=fpn_feature_channels, out_channels=aspp_channels)

        self.feature_projections = []
        self.feature_merge = []

        for i in range(self.decoder_stages):
            self.feature_projections.append(
                nn.Sequential(
                    nn.Conv2d(low_level_channels[i], low_level_project_channels[i], 1, bias=False),
                    nn.BatchNorm2d(low_level_project_channels[i]),
                    nn.ReLU())
            )

            if i == 0:
                merged_channels = aspp_channels + low_level_project_channels[i]
            else:
                merged_channels = decoder_channels + low_level_project_channels[i]

            self.feature_merge.append(
                DSConv(merged_channels, decoder_channels, kernel_size=5, padding=2)
            )
        self.feature_projections = torch.nn.ModuleList(self.feature_projections)
        self.feature_merge = torch.nn.ModuleList(self.feature_merge)


    def forward(self, features):

        x = features[self.fpn_feature_key]
        x = self.aspp(x)

        for i in range(self.decoder_stages):
            f = features[self.fpn_low_level_feature_keys[i]]
            f = self.feature_projections[i](f)
            x = F.interpolate(x, size=f.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat((x, f), dim=1)
            x = self.feature_merge[i](x)

        return x


class DeepLabPredHead(nn.Module):

    def __init__(self,
                 in_channels: int,
                 internal_channels: int,
                 out_channels: int,
                 key_name: str = None):
        super(DeepLabPredHead, self).__init__()
        self.feature_conv = DSConv(in_channels, internal_channels, kernel_size=5, padding=2)
        self.output = nn.Conv2d(internal_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.key_name = key_name


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_conv(x)
        x = self.output(x)

        if self.key_name:
            x = {self.key_name: x}

        return x


if __name__ == '__main__':

    fpn_features = {
        "k1": torch.rand(1, 3, 4, 4),
        "k2": torch.rand(1, 4, 8, 8),
        "k3": torch.rand(1, 5, 16, 16)
    }

    decoder = DeepLabEncoder(fpn_feature_key="k1",
                             fpn_feature_channels= 3,
                             fpn_low_level_feature_keys=["k2", "k3"],
                             low_level_channels=[4, 5],
                             low_level_project_channels=[4, 2])

    out = decoder.forward(fpn_features)

    print(out.shape)
    assert out.shape == (1, 256) + fpn_features["k3"].shape[2:]

    pred_head = DeepLabPredHead(in_channels=256, internal_channels=64, out_channels=2)

    pred_out = pred_head(out)

    print(pred_out.shape)
    assert pred_out.shape == (1, 2) + fpn_features["k3"].shape[2:]

    pred_head = DeepLabPredHead(in_channels=256, internal_channels=128, out_channels=1)

    pred_out = pred_head(out)

    print(pred_out.shape)
    assert pred_out.shape == (1, 1) + fpn_features["k3"].shape[2:]


