from CGAN_train import CGAN
from pytorch_lightning import Trainer
import os
import cv2
from data.dataset import create_dataloader
import matplotlib.pyplot as plt
import numpy as np

import torch


def eval_model(path: str, model_name: str, input_data_path: str, target_data_path: str, img_type="tif"):
    model = CGAN(path)
    checkpoint = torch.load(f"{path}/{model_name}.ckpt")
    model.load_state_dict(checkpoint['state_dict'])
    epoch_idx = checkpoint['epoch']
    model.cuda()
    model.eval()
    dataload = create_dataloader(path_a=input_data_path, path_b=target_data_path, split=0.896,
                                 train=False, crop_size=(1024, 1024), batch_size=1, repeat=0)
    if not os.path.isdir(path):
        os.mkdir(path)
    if not os.path.isdir(img_path):
        os.mkdir(img_path)
    pred_path = f"{img_path}/preds"
    if not os.path.isdir(pred_path):
        os.mkdir(pred_path)
    targ_path = f"{img_path}/targs"
    if not os.path.isdir(targ_path):
        os.mkdir(targ_path)
    sums_path = f"{img_path}/sums"
    if not os.path.isdir(sums_path):
        os.mkdir(sums_path)
    print(epoch_idx)
    batch_idx = 0
    for i, img in enumerate(dataload):
        # Unscaled images
        if img_type == "tif":
            batch_idx = batch_idx + 1
            input, target = img
            cuda_input = input.cuda()
            pred = model(cuda_input)
            input = np.exp(input.squeeze().numpy())
            pred = np.exp(pred.detach().to('cpu').squeeze().numpy())
            target = np.exp(target.squeeze().numpy())
            filename = f"{pred_path}/{batch_idx}_Test_pred.tif"
            cv2.imwrite(filename, pred)
            filename = f"{targ_path}/{batch_idx}_Test_target.tif"
            cv2.imwrite(filename, target)
            filename = f"{sums_path}/{batch_idx}_Test_input.tif"
            cv2.imwrite(filename, input)
        # Scaling images to 0-255
        if img_type == "png":
            batch_idx = i + 1
            input, target = img
            cuda_input = input.cuda()
            pred = model(cuda_input)
            input = np.exp(input.squeeze().numpy())
            pred = np.exp(pred.detach().to('cpu').squeeze().numpy())
            target = np.exp(target.squeeze().numpy())
            filename = f"{pred_path}/{batch_idx}_Test_pred.png"
            cv2.imwrite(filename, pred * (255 / pred.max()))
            filename = f"{targ_path}/{batch_idx}_Test_target.png"
            cv2.imwrite(filename, target * (255 / target.max()))
            filename = f"{sums_path}/{batch_idx}_Test_input.png"
            cv2.imwrite(filename, input * (255 / input.max()))
        else:
            raise NameError("Choose 'png' or 'tif' as img_type")
# path = f"/media/data1/projsweep3D/minval/cgan_osa/"
path = f"/media/data1/projsweep3D/maxproj_1250"
# model_name = "min_val_lightning_model"
model_name = "min_val"
img_type = "tif"
img_path = f"{path}/ms_images_rc"
if not os.path.isdir(path):
    os.mkdir(path)
if not os.path.isdir(img_path):
    os.mkdir(img_path)

path_a = "/media/data1/projsweep3D/201029_3D_FLR_incucyte/IC12_VID1260_single spheroid/c2sumsweepAligned"
path_b = "/media/data1/projsweep3D/201029_3D_FLR_incucyte/IC12_VID1260_single spheroid/C2maxproj"

# path_a = "/media/data1/projsweep3D/deconvolutions/2021_1026_deconvolutions/IC12_VID1260_single spheroid/c2sweeps_RLTV_maxproj"
# path_b = "/media/data1/projsweep3D/deconvolutions/2021_1026_deconvolutions/IC12_VID1260_single spheroid/c2sweeps_RLTV_oqcproj"

# path_a = "/media/data1/projsweep3D/deconvolutions/2021_1026_deconvolutions/IC12_VID1264_single plane MS/c2sweeps_RLTV_maxproj"
# path_b = "/media/data1/projsweep3D/deconvolutions/2021_1026_deconvolutions/IC12_VID1264_single plane MS/c2sweeps_RLTV_oqcproj"

path_a = "/media/data1/projsweep3D/201029_3D_FLR_incucyte/IC12_VID1264_single plane MS/c2sumsweepAligned"
path_b = "/media/data1/projsweep3D/201029_3D_FLR_incucyte/IC12_VID1264_single plane MS/C2oqcProj"

eval_model(path=path, model_name=model_name, input_data_path=path_a, target_data_path=path_b)
















