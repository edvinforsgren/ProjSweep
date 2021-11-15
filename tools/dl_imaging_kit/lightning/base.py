from typing import *

import torch
from torch import nn
from torch import optim
from pytorch_lightning import LightningModule
from kornia import losses

from dl_imaging_kit import metrics


class StandardSupervisedModule(LightningModule):

    def __init__(self):
        super(StandardSupervisedModule, self).__init__()
        self._net = self.net()
        self._loss = self.loss()
        self._metrics = self.metrics()
        self._learning_rate = self.learning_rate()

    def net(self) -> nn.Module:
        raise NotImplementedError('Override net() to create your model instance.')

    def loss(self) -> Callable:
        raise NotImplementedError('Override loss() to return your loss callable.')

    def learning_rate(self) -> float:
        """ Override to return default learning-rate. """
        return 1e-3

    def metrics(self) -> Sequence[Callable]:
        """ Override to return evaluation metrics. """
        return []

    def forward(self, x: torch.Tensor):
        return self._net(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_nb: int) -> Dict:
        images, target = batch
        pred = self.forward(images)
        loss = self._loss(pred, target)

        log = {
            'train_loss': loss.item()
        }

        return {'loss': loss, 'log': log}

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_nb: int) -> Dict[str, torch.Tensor]:
        scores = self.evaluate_batch(batch)
        return {'val_loss': scores['loss'], 'log': {f'val_{name}': score for name, score in scores.items()}}

    def validation_epoch_end(
            self,
            outputs: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        return self._aggregate_scores(outputs, 'val')

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_nb: int) -> Dict[str, torch.Tensor]:
        scores = self.evaluate_batch(batch)
        return {'test_loss': scores['loss'], 'log': {f'test_{name}': score for name, score in scores.items()}}

    def test_epoch_end(
            self,
            outputs: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        return self._aggregate_scores(outputs, 'test')

    def _aggregate_scores(self, outputs: List[Dict], prefix: str):
        loss_name = f'{prefix}_loss'
        avg_loss = torch.tensor([x[loss_name] for x in outputs]).mean()
        log = {loss_name: avg_loss}
        for metric_name in outputs[0]['log'].keys():
            avg_metric = torch.tensor([x['log'][metric_name] for x in outputs]).mean()
            log[metric_name] = avg_metric
        return {loss_name: avg_loss, 'log': log}

    def evaluate_batch(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        images, target = batch
        pred = self.forward(images)
        loss = self._loss(pred, target).item()

        metric_scores = {'loss': loss}
        for metric in self._metrics:
            metric_score = metric(pred, target)
            name = metric.__name__ if hasattr(metric, '__name__') else type(metric).__name__
            metric_scores[name] = metric_score

        return metric_scores


class SemanticSegmentationModule(StandardSupervisedModule):

    def loss(self) -> Callable:
        return losses.DiceLoss()

    def metrics(self) -> Sequence[Callable]:
        return (metrics.pixelwise_accuracy, metrics.pixelwise_f1score)

    def configure_optimizers(self) -> Sequence[optim.Optimizer]:
        optimizer = optim.Adam(self.parameters(), lr=self._learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         patience=10,
                                                         min_lr=1e-6)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
