from pytorch_lightning import LightningModule
from torch.nn import L1Loss, MSELoss
from torch import optim
import torch
from typing import *
from dl_imaging_kit.utils.pan_post_process import process_panoptic_segmentation
from dl_imaging_kit.models.panoptic_deeplab import PanopticDeepLab
from dl_imaging_kit.losses import RegularCE
from dl_imaging_kit.models.encoder import VovNetEncoder

from pytorch_lightning import Trainer
from dl_imaging_kit.dataset import PanopticDeepDataset
from torch.utils.data.dataloader import DataLoader

class PanopticDeeplabModule(LightningModule):

    def __init__(self):

        super().__init__()
        self.backbone = VovNetEncoder(in_channels=1,
                                      channels_per_depth=(32, 64, 128),
                                      convs_per_depth=(5, 5, 5))
        self._learning_rate = 0.00005

        out_feature = 'block_3'
        low_lvl_feature_keys = ['block_2', 'block_1']

        self.panoptic_deeplab_model = PanopticDeepLab(fpn_feature_key=out_feature, fpn_feature_channels=128,
                                                      fpn_low_level_feature_keys=low_lvl_feature_keys,
                                                      low_level_channels=[64, 32],
                                                      backbone=self.backbone)

        self.semantic_loss = RegularCE(ignore_label=255)
        self.semantic_loss_weight = 1.0
        self.center_loss = MSELoss()
        self.center_loss_weight = 200.0
        self.offset_loss = L1Loss()
        self.offset_loss_weight = 0.01
        self._loss = self.loss

    def loss(self, prediction, target):

        loss = 0
        log = {}
        if "semantic" in prediction:
            sem_loss = self.semantic_loss(prediction["semantic"], target['semantic'])
            if self.semantic_loss_weight is not None:
                sem_loss = sem_loss * self.semantic_loss_weight
            log["sem_loss"] = sem_loss
            loss += sem_loss

        if "center" in prediction:
            center_loss = self.center_loss(prediction["center"], target["center"])
            if self.center_loss_weight is not None:
                center_loss = center_loss * self.center_loss_weight
            log["center_loss"] = center_loss
            loss += center_loss

        if "offset" in prediction:
            offset_loss = self.offset_loss(prediction["offset"], target["offset"])
            if self.offset_loss_weight is not None:
                offset_loss = offset_loss * self.offset_loss_weight
            log["offset_loss"] = offset_loss
            loss += offset_loss
        log["loss"] = loss

        return loss, log

    def forward(self, x):
        return self.panoptic_deeplab_model(x)

    def interpolate(self, x):
        out = self.panoptic_deeplab_model(x)
        return process_panoptic_segmentation(out)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_nb: int) -> Dict:
        images, target = batch
        pred = self.forward(images)
        (loss, log) = self._loss(pred, target)


        return {'loss': loss, 'log': log}

    def configure_optimizers(self):

        optimizer = optim.Adam(self.parameters(), lr=self._learning_rate)
        #Create a warmup scheduler
        warmup_iters = 1000
        warmup_factor = 1.0 / 1000
        lambda_warmup = lambda epoch: warmup_factor * ((warmup_iters - epoch) / warmup_iters)

        steps = (30000,)
        gamma = 0.1

        scheduler_1 = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_warmup, last_epoch=warmup_iters)
        scheduler_2 = optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma)
        return {'optimizer': optimizer, 'lr_scheduler': [scheduler_1, scheduler_2]}


    def train_dataloader(self) -> DataLoader:

        img_path = "/home/data/cell_data/images/test_phase"
        target_path = "/home/data/cell_data/images/deeplab_pan/test"
        dataset = PanopticDeepDataset(img_path, target_path)
        return DataLoader(dataset, batch_size=2)


if __name__ == "__main__":

    model = PanopticDeeplabModule()
    trainer = Trainer(gpus=1, max_epochs=20, num_nodes=1)   # , LrfinderCallback()])
