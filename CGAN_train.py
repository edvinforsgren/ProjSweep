"""
Created on 2020-07-01. 14:05
Last edited 2021-11-15
@authors: Christoffer Edlund and Edvin Forsgren

"""
import os
import torch
from typing import *
import sys
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
from pytorch_lightning import Trainer
import dl_imaging_kit as dlik
import numpy as np
from data.dataset import create_dataloader
import cv2
import pix2pix_networks as p2p
from dl_imaging_kit.lightning.gan import CGANModule
import time

sys.path.append("/home/edfo/local_libs/tools/")


class CGAN(CGANModule):
    def __init__(self, path=None, path_a=None, path_b=None, gen_type="OSA"):
        super(CGAN, self).__init__()
        self.l1_loss_weight = 10
        self.l1_weight = 10
        self.preds = []
        self.sources = []
        self.targets = []
        self.generator = self.Generator()
        self.discriminator = self.Discriminator()
        self.vloss = None
        self.path = path
        self.path_a = path_a
        self.path_b = path_b
        self.optidx = None
        self.gen_type = gen_type

    def l1_loss_weight(self):
        return 10

    def Generator(self) -> torch.nn.Module:
        model = dlik.models.unet.UNet(in_channels=1,
                                      out_channels=1,
                                      convs_per_depth=(5, 5, 5, 5),
                                      channels_per_depth=(32, 64, 128, 256),
                                      block_class=dlik.models.blocks.OSABlock,  # dlik.models.blocks.ConvBlock)
                                      block_kwargs={
                                          'squeeze_and_excitation_block': dlik.models.blocks.EfficientSqueezeAndExcitation})
        return model

    def Discriminator(self) -> torch.nn.Module:
        model = p2p.define_D(2, 64, netD="basic")
        return model

    def loss(self) -> Callable:
        return F.mse_loss

    def loss_l1(self) -> Callable:
        return F.l1_loss

    def learning_rate(self) -> Union[float, Sequence[float]]:
        lr = 2e-4
        return lr

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_nb: int, optimizer_idx: int) -> Dict:
        train_x, train_y = batch
        epoch_idx = self.current_epoch
        self.optidx = optimizer_idx
        if optimizer_idx == 0:
            fake_y = self.forward(train_x)

            self.fake_y = fake_y

            disc_input_fake = torch.cat([train_x, self.fake_y], dim=1)  # Combine real input with fake output
            l1_loss = self.loss_l1()(train_y, self.fake_y)

            predicton = self.discriminator(disc_input_fake)

            label_real = torch.ones(predicton.size())

            if self.on_gpu:
                label_real = label_real.cuda(train_x.device)

            loss_g = self.loss()(predicton, label_real)

            loss = loss_g + l1_loss * self.l1_weight
            log = {"loss_G": loss_g.item(), "l1_loss_g": l1_loss.item()}
            # if self.logger is not None:
            #     self.logger.log_metrics(log)
            # output = OrderedDict({'loss': loss, "log": log})
            output = {'loss': loss}
            return output

        if optimizer_idx == 1:  # and epoch_idx > 50:
            fake_y = self.forward(train_x)

            disc_input_real = torch.cat([train_x, train_y], dim=1)  # Combine real input with real output
            disc_input_fake = torch.cat([train_x, fake_y.detach()], dim=1)  # Combine real input with fake output

            real_prediction = self.discriminator(disc_input_real)
            fake_prediction = self.discriminator(disc_input_fake)

            label_real = torch.ones(real_prediction.size())
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
            # if self.logger is not None:
            #     self.logger.log_metrics(log)
            # output = OrderedDict({'loss': loss_D, "log": log})
            output = {'loss': loss_D}
            return output

    def training_epoch_end(self, outputs):
        # Save the training loss
        train_loss_mean = 0
        i = 0
        for output in outputs:
            for op in output:
                loss = op["loss"]
                loss = torch.mean(loss)
                train_loss_mean += loss
                i += 1
        train_loss_mean /= i
        txtpath = f"{self.path}/train_loss.txt"
        f = open(txtpath, "a")
        f.write(f"{loss.item()} \n")
        f.close()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        fake_y = self.generator(x)
        loss = F.l1_loss(fake_y, y)
        output = {'val_loss': loss}
        return output

    def validation_epoch_end(self, outputs):
        # Save all validation losses
        val_loss_mean = 0
        for output in outputs:
            val_loss = output["val_loss"]
            val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss
        val_loss_mean /= len(outputs)
        txtpath = f"{self.path}/val_loss.txt"
        f = open(txtpath, "a")
        f.write(f"{val_loss_mean.item()} \n")
        f.close()

        # Save first minimum validation model
        if self.vloss is None:
            self.vloss = val_loss_mean.item()
            val_model_path = f"{self.path}/min_val"
            trainer.save_checkpoint(val_model_path + ".ckpt")

        # Save a bunch of models for evaluation purposes
        if (self.current_epoch % 50) == 0:
            epoch_path = f"{self.path}/epoch_models"
            if not os.path.isdir(epoch_path):
                os.mkdir(epoch_path)
            trainer.save_checkpoint(f"{epoch_path}/{self.current_epoch}_model.ckpt")

        # Update the model to the new minimum validation loss model
        if self.vloss > val_loss_mean.item():
            self.vloss = val_loss_mean.item()
            val_model_path = f"{self.path}/min_val"
            trainer.save_checkpoint(val_model_path + ".ckpt")

    def test_step(self, batch, batch_idx):
        x, y = batch
        print_frq = 1
        batch_idx = batch_idx + 1
        epoch_idx = self.current_epoch
        img_path = f"{self.path}/images/"
        if not os.path.isdir(img_path):
            os.mkdir(img_path)
        if batch_idx % print_frq == 0:
            y_hat = self.generator(x)
            test_pred_img = y_hat.detach().to('cpu').numpy()
            test_in_img = x.detach().to('cpu').numpy()
            test_out_im = y.detach().to('cpu').numpy()
            image = np.exp(test_pred_img)
            scale = (255 / np.max(image))
            image = np.squeeze(image)
            image = image * scale
            filename = f"{img_path}{batch_idx:03d}_{epoch_idx}_Test_pred.png"
            cv2.imwrite(filename, image)

            image = np.exp(test_in_img)
            image = np.squeeze(image)
            filename = f"{img_path}{batch_idx:03d}_{epoch_idx}_Test_input.png"
            cv2.imwrite(filename, image * (255 / image.max()))

            image = np.exp(test_out_im)
            image = np.squeeze(image)
            image = image * scale
            filename = f"{img_path}{batch_idx:03d}_{epoch_idx}_Test_target.png"
            cv2.imwrite(filename, image)

    def train_dataloader(self) -> DataLoader:
        # path_a = self.input_path #  "/media/data1/projsweep3D/201029_3D_FLR_incucyte/IC12_VID1264_single plane MS/c2sumsweepAligned"
        # path_b = self.target_path #  "/media/data1/projsweep3D/201029_3D_FLR_incucyte/IC12_VID1264_single plane MS/C2oqcProj"
        return create_dataloader(self.path_a, self.path_b, batch_size=2, split=0.896, on_gpu=self.on_gpu)

    def val_dataloader(self):
        # path_a = self.input_path  # "/media/data1/projsweep3D/201029_3D_FLR_incucyte/IC12_VID1264_single plane MS/c2sumsweepAligned"
        # path_b = self.target_path  # "/media/data1/projsweep3D/201029_3D_FLR_incucyte/IC12_VID1264_single plane MS/C2oqcProj"
        return create_dataloader(self.path_a, self.path_b, batch_size=1, split=0.896, crop_size=(512, 512), train=False,
                                 on_gpu=self.on_gpu)

    def test_dataloader(self):
        # path_a = self.input_path #  "/media/data1/projsweep3D/201029_3D_FLR_incucyte/IC12_VID1264_single plane MS/c2sumsweepAligned"
        # path_b = self.target_path #  "/media/data1/projsweep3D/201029_3D_FLR_incucyte/IC12_VID1264_single plane MS/C2oqcProj"
        return create_dataloader(self.path_a, self.path_b, train=False, batch_size=1, split=0.896, crop_size=(512, 512),
                                 on_gpu=self.on_gpu)


if __name__ == "__main__":
    path = "/media/data1/projsweep3D/test_dir"
    if not os.path.isdir(path):
        os.mkdir(path)

    input_path = "/media/data1/projsweep3D/201029_3D_FLR_incucyte/IC12_VID1264_single plane MS/c2sumsweepAligned"
    target_path = "/media/data1/projsweep3D/201029_3D_FLR_incucyte/IC12_VID1264_single plane MS/C2oqcProj"
    max_epochs = 4
    num_nodes = 1
    gpus = -1  # -1 -> Train on all available GPUs
    accelerator = "ddp"  # Generally fastest when training on GPU. Details: https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html
    default_root_dir = f"{path}/checkpoints"
    generator_type = "OSA"
    model = CGAN(path=path, path_a=input_path, path_b=target_path, gen_type=generator_type)
    trainer = Trainer(gpus=gpus, accelerator=accelerator, max_epochs=max_epochs, num_nodes=num_nodes, logger=None,
                      default_root_dir=f"{path}/checkpoints")
    t = time.time()
    trainer.fit(model)
    print("Training time:", time.time() - t)
    trainer.test()
    trainer.save_checkpoint(f"{path}/end_model.ckpt")
    model_path = f"{path}/checkpoints/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_name = "/model"
    model_path = model_path + model_name
    trainer.save_checkpoint(model_path + ".ckpt")
    torch.save(model.generator, model_path + "_generator" + ".pth")
    torch.save(model.discriminator, model_path + "_discriminator" + ".pth")
