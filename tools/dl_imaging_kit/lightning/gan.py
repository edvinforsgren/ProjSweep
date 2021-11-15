import torch
from torch import nn
from typing import *
from collections import OrderedDict
from pytorch_lightning import LightningModule
from torch import functional as F

class StandardGANModule(LightningModule):

    def __init__(self):

        super(StandardGANModule, self).__init__()
        self.generator = self.Generator()
        self.discriminator = self.Discriminator()
        self._loss = self.loss()
        self._metrics = self.metrics()
        self._learning_rate = self.learning_rate()
        self.fake_y = None

    @property
    def latent_dim(self) -> Sequence[int]:
        raise NotImplementedError('Override latent_dim() to define the size of the generator input.')

    @property
    def optimizer(self) -> Union[torch.optim.Optimizer, Sequence[torch.optim.Optimizer]]:
        return NotImplementedError('Override optimizer() to define the optimizers for the generator and discriminator,'
                                   'either as a optimizer or a sequence with '
                                   '[generator_optimizer, discriminator_optimizer]')

    def Generator(self) -> nn.Module:
        raise NotImplementedError('Override Generator() to create your generator instance.')

    def Discriminator(self) -> nn.Module:
        raise NotImplementedError('Override Discriminator() to create your discriminator instance.')

    def loss(self) -> Callable:
        raise NotImplementedError('Override loss() to return your loss callable.')

    def learning_rate(self) -> Union[float, Sequence[float]]:
        """ Override to return default learning-rate. """
        return 1e-3

    def metrics(self) -> Sequence[Callable]:
        """ Override to return evaluation metrics. """
        return []

    def forward(self, x: torch.Tensor):
        return self.generator(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_nb: int, optimizer_idx: int) -> Dict:
        images, _ = batch

        if optimizer_idx == 0:

            z = torch.randn(images.shape[0], self.latent_dim)
            fake_y = self.forward(z)
            self.fake_y = torch.tanh(fake_y)
            label_real = torch.ones(images.size(0), 1)

            if self.on_gpu:
                label_real = label_real.cuda(images.device)

            loss_G = self._loss(self.discriminator(self.fake_y), label_real)
            log = {"loss_G": loss_G}

            output = OrderedDict({'loss': loss_G, "log":log})

            return output

        if optimizer_idx == 1:

            label_real = torch.ones(images.size(0), 1)
            label_fake = torch.zeros(images.size(0), 1)
            if self.on_gpu:
                label_real = label_real.cuda(images.device)
                label_fake = label_fake.cuda(images.device)

            loss_real = self.adv_loss(self.discriminator(images), label_real)
            loss_fake = self.adv_loss(self.discriminator(self.fake_y.detach()), label_fake)

            loss_D = (loss_real + loss_fake) / 2
            log = {"loss_D": loss_D.item(), "loss_D_fake": loss_fake.item(),
                                         "loss_D_real": loss_real.item()}

            output = OrderedDict({'loss': loss_D, "log":log})

            return output


class CGANModule(StandardGANModule):

    def loss(self) -> Callable:
        return torch.nn.L1Loss()

    @property
    def optimizer(self) -> Union[torch.optim.Optimizer, Sequence[torch.optim.Optimizer]]:
        return torch.optim.Adam

    def configure_optimizers(self) -> Sequence[torch.optim.Optimizer]:

        lr = self.learning_rate()
        optimizer = self.optimizer

        if not isinstance(lr, Sequence):
            lr = [lr]
        if not isinstance(optimizer, Sequence):
            optimizer = [optimizer]

        if len(lr) == 1:
            lr_G = lr_D = lr[0]
        elif len(self.lr) == 2:
            lr_G = lr[0]
            lr_D = lr[1]

        if len(optimizer) == 1:
            opt_G = optimizer[0](self.parameters(), lr_G)
            opt_D = optimizer[0](self.parameters(), lr_D)
        elif len(self.optim) == 2:
            opt_G = optimizer[0](self.parameters(), lr_G)
            opt_D = optimizer[2](self.parameters(), lr_D)
        if self.logger:
            self.logger.log_hyperparams({"lr_gen": lr_G, "lr_disc": lr_D})

        return {'optimizer': [opt_G, opt_D]}


class CGANModule(StandardGANModule):

    def __init__(self):
        super(CGANModule, self).__init__()
        self.l1_weight = self.l1_loss_weight()

    def l1_loss_weight(self):
        raise NotImplementedError('Override l1_loss_weight to return the l1 loss wight, 10 is a good default')

    def Generator(self) -> torch.nn.Module:
        raise NotImplementedError('Override Generator to create your discriminator instance.')

    def Discriminator(self) -> torch.nn.Module:
        raise NotImplementedError('Override Discriminator to create your discriminator instance.')

    @property
    def optimizer(self) -> Union[torch.optim.Optimizer, Sequence[torch.optim.Optimizer]]:
        return torch.optim.Adam

    @property
    def soft_labels(self):
        return True

    def loss(self) -> Callable:
        raise NotImplementedError('Override loss to define the loss function, MSE might be a good choice')

    def loss_l1(self) -> Callable:
        return F.l1_loss

    def learning_rate(self) -> Union[float, Sequence[float]]:
        """ Override to return default learning-rate. """

        raise NotImplementedError('Override learning_rate to define the learning rate.')


    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_nb: int, optimizer_idx: int) -> Dict:
        train_x, train_y = batch

        if optimizer_idx == 0:

            fake_y = self.forward(train_x)
            self.fake_y = torch.tanh(fake_y)  # added tanh activation, should probably make the activation an option


            disc_input_fake = torch.cat([train_x, self.fake_y], dim=1)   # Combine real input with fake output
            l1_loss = self.loss_l1()(train_y, self.fake_y)

            predicton = self.discriminator(disc_input_fake)
            label_real = torch.ones(predicton.size())

            if self.on_gpu:
                label_real = label_real.cuda(train_x.device)

            loss_g = self.loss()(predicton, label_real)

            loss = loss_g + l1_loss * self.l1_weight
            log = {"loss_G": loss_g, "l1_loss_g": l1_loss}

            output = OrderedDict({'loss': loss, "log": log})

            return output

        if optimizer_idx == 1:
            fake_y = self.forward(train_x)
            fake_y = torch.tanh(fake_y)  # added tanh activation, should probably make the activation an option

            disc_input_real = torch.cat([train_x, train_y], dim=1)  # Combine real input with real output
            disc_input_fake = torch.cat([train_x, fake_y.detach()], dim=1)  # Combine real input with fake output

            real_prediction = self.discriminator(disc_input_real)
            fake_prediction = self.discriminator(disc_input_fake)

            label_real = torch.ones(real_prediction.size())

            if self.soft_labels:
                label_real = label_real - torch.rand(label_real.size()) * 0.1  # Creating soft labels


            label_fake = torch.zeros(fake_prediction.size())

            if self.on_gpu:
                label_real = label_real.cuda(train_x.device)
                label_fake = label_fake.cuda(train_x.device)

            loss_real = self.loss()(real_prediction, label_real)
            loss_fake = self.loss()(fake_prediction, label_fake)

            loss_D = (loss_real + loss_fake) / 2
            log = {"loss_D": loss_D.item(), "loss_D_fake": loss_fake.item(),
                                         "loss_D_real": loss_real.item()}

            output = OrderedDict({'loss': loss_D, "log": log})

            return output

    def configure_optimizers(self) -> Sequence[torch.optim.Optimizer]:

        lr = self.learning_rate()
        optimizer = self.optimizer

        if not isinstance(lr, Sequence):
            lr = [lr]
        if not isinstance(optimizer, Sequence):
            optimizer = [optimizer]

        if len(lr) == 1:
            lr_G = lr_D = lr[0]
        elif len(lr) == 2:
            lr_G = lr[0]
            lr_D = lr[1]
        else:
            raise ValueError(f"Was looking for one or two lr's, got: {len(lr)}")

        if len(optimizer) == 1:
            opt_G = optimizer[0](self.parameters(), lr_G)
            opt_D = optimizer[0](self.parameters(), lr_D)
        elif len(self.optim) == 2:
            opt_G = optimizer[0](self.parameters(), lr_G)
            opt_D = optimizer[2](self.parameters(), lr_D)
        if self.logger:
            self.logger.log_hyperparams({"lr_gen": lr_G, "lr_disc": lr_D})

        return [opt_G, opt_D], []

