import os
import sys
from pathlib import Path
import cv2
from sklearn.metrics import r2_score
import numpy as np
sys.path.append("/home/edfo/local_libs/tools/")
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
import pytorch_fid.fid_score

m_path = "/media/data1/projsweep3D"

def fid_calc():
    path_a = f"{m_path}/minval/cgan_osa/tifs/ocq/"
    nets = ["cgan_osa", "cgan_conv", "UNet_osa", "UNet_conv"]
    for net in nets:
        path_b = f"{m_path}/minval/{net}/tifs/preds/"
        fid_value = pytorch_fid.fid_score.calculate_fid_given_paths([path_a, path_b], 100, device=0, dims=2048)
        print(f"Fid for {net}: {fid_value}")
# fid_calc()


# def tophat():
#     nets = ["cgan_osa", "cgan_conv", "UNet_osa", "UNet_conv"]
#     for net in nets:
#         p_dir = f"C:\\Users\\edfo0007\\Documents\\images\\ssa_{net}_output\\tophat_png"
#         path_b = Path(f"C:\\Users\\edfo0007\\Documents\\images\\ssa_{net}_output\\tifsv2")
#         files = list(path_b.glob('*.tif'))
#         if not os.path.isdir(p_dir):
#             os.mkdir(p_dir)
#         for f in files:
#             img = tbr(cv2.imread(str(f), -1), morph_size=(30, 30))
#             cv2.imwrite(f"{p_dir}\\{os.path.basename(str(f))[:-3]}.png", img * (255 / img.max()))


# path_b = os.path.join(r"C:\Users\edfo0007\Documents\images\201029_3D_FLR_incucyte\IC12_VID1260_single spheroid\c2sumsweepAligned")
# fid_value = pytorch_fid.fid_score.calculate_fid_given_paths([path_a, path_b], 96, True, 2048)
# print(f"Fid for input vs output: {fid_value}")


def crop():
    nets = ["cgan_osa", "cgan_conv", "UNet_osa", "UNet_conv"]
    h = 500
    w = 500
    for net in nets:
        path_b = Path(f"C:\\Users\\edfo0007\\Documents\\images\\ssa_{net}_output\\tifs")
        dirname = f"C:\\Users\\edfo0007\\Documents\\images\\ssa_{net}_output\\cropped_tifs"
        files = list(path_b.glob('*.tif'))
        i = 1
        for f in files:
            image = cv2.imread(str(f), -1)
            ch, cw = image.shape

            x = cw / 2 - w / 2
            y = ch / 2 - h / 2
            crop_img = image[int(y):int(y + h), int(x):int(x + w)]
            if not os.path.isdir(dirname):
                os.mkdir(dirname)
            cv2.imwrite(f"{dirname}\\00{i}_cropped.tif", crop_img)
            i = i + 1


# fid_calc()


def scale():
    nets = ["cgan_osa", "cgan_conv", "UNet_osa", "UNet_conv"]
    for net in nets:
        path_b = Path(f"C:\\Users\\edfo0007\\Documents\\images\\ssa_{net}_output\\cropped_tifs")
        dirname = f"C:\\Users\\edfo0007\\Documents\\images\\ssa_{net}_output\\cropped_pngs"
        files = list(path_b.glob('*.tif'))
        i = 1
        for f in files:
            image = cv2.imread(str(f), -1)
            scale_img = image * (255 / image.max())
            if not os.path.isdir(dirname):
                os.mkdir(dirname)
            cv2.imwrite(f"{dirname}\\00{i}_cropped.png", scale_img)
            i = i + 1


def rsq():
    nets = ["cgan_osa", "cgan_conv", "UNet_osa", "UNet_conv"]
    for net in nets:
        path = f"C:\\Users\\edfo0007\\Documents\\images\\ssa_{net}_output\\crop_int"
        otot = np.load(f"{path}\\oqc-intens-array.npy")
        ptot = np.load(f"{path}\\pred-intens-array.npy")
        r2 = r2_score(np.log(otot), np.log(ptot))
        print(f"R^2 for {net}: {r2}")


def PSNR():
    nets = ["cgan_osa", "cgan_conv", "UNet_osa", "UNet_conv"]
    targs = Path(f"{m_path}/minval/cgan_osa/tifs/ocq/")
    targs = list(targs.glob('*.tif'))
    targs.sort()
    all_psnr = []
    for net in nets:
        # preds = Path(f"C:\\Users\\edfo0007\\Documents\\images\\ssa_{net}_output\\tifsv2")
        # preds = Path(f"C:\\Users\\edfo0007\\Documents\\images\\ss_{net}_cropped")
        preds = Path(f"{m_path}/minval/{net}/tifs/preds/")
        # targs = Path(f"C:\\Users\\edfo0007\\Documents\\images\\201029_3D_FLR_incucyte\\IC12_VID1260_single spheroid\\C2oqcProj")
        # preds = Path(f"C:\\Users\\edfo0007\\Documents\\images\\201029_3D_FLR_incucyte\\IC12_VID1260_single spheroid\\c2sumsweepAligned")
        # targs = Path(f"C:\\Users\\edfo0007\\Documents\\images\\201029_3D_FLR_incucyte\\IC12_VID1264_single plane MS\\C2oqcProj")

        # targs = Path(f"C:\\Users\\edfo0007\\Documents\\images")
        preds = list(preds.glob('*.tif'))
        preds.sort()
        psnrs = []
        ssims = []
        for pred, targ in zip(preds, targs):
            original = cv2.imread(str(targ), -1)
            compressed = cv2.imread(str(pred), -1)
            ssims.append(ssim(original, compressed, gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
            # print(compressed)
            mse = np.mean((original - compressed) ** 2)
            if (mse == 0):  # MSE is zero means no noise is present in the signal .
                # Therefore PSNR have no importance.
                print("MSE = 0")
            max_pixel = 255.0
            psnrs.append(20 * np.log10(max_pixel / np.sqrt(mse)))
        mpsnr = np.mean(psnrs)
        stdpsnr = np.std(psnrs)
        mssim = np.mean(ssims)
        stdssim = np.std(ssims)
        print(f"For {net}: mean PSNR: {mpsnr} dB, std PSNR: {stdpsnr} dB \n SSIM: {mssim}, std ssim: {stdssim}")
        all_psnr.append(psnrs)
    return all_psnr


# psnrs = PSNR()
# print(psnrs)
# for psnr in psnrs:
#     print(min(psnr), psnr.index(min(psnr)))


def ss_cropper():
    targs = Path(
        f"C:\\Users\\edfo0007\\Documents\\images\\201029_3D_FLR_incucyte\\IC12_VID1260_single spheroid\\C2oqcProj")
    nets = ["sums"]  # ["cgan_osa", "cgan_conv", "UNet_osa", "UNet_conv"]
    i = 0
    targs = list(targs.glob('*.tif'))
    for net in nets:
        predis = Path(f"C:\\Users\\edfo0007\\Documents\\images\\ssa_{net}_output\\tifsv2")
        predis = Path(
            f"C:\\Users\\edfo0007\\Documents\\images\\201029_3D_FLR_incucyte\\IC12_VID1260_single spheroid\\c2sumsweepAligned")
        pathstofiles = f"C:\\Users\\edfo0007\\Documents\\images"
        preds = list(predis.glob('*.tif'))
        for targ, pred in zip(targs, preds):
            if not os.path.basename(pred).__contains__("E"):
                image = cv2.imread(str(targ), -1)
                thresh = threshold_otsu(image)
                bw = closing(image > thresh, square(3))

                # remove artifacts connected to image border
                cleared = clear_border(bw)

                # label image regions
                label_image = label(cleared)
                # to make the background transparent, pass the value of `bg_label`,
                # and leave `bg_color` as `None` and `kind` as `overlay`
                # image_label_overlay = label2rgb(label_image, image=image, bg_label=0)

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(image, 'gray')

                for region in regionprops(label_image):
                    # take regions with large enough areas
                    if region.area >= 500:
                        # draw rectangle around segmented coins
                        minr, minc, maxr, maxc = region.bbox
                        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                                  fill=False, edgecolor='red', linewidth=2)
                        ax.add_patch(rect)
                        r_cen, c_cen = region.centroid
                        r_cen, c_cen = int(r_cen), int(c_cen)
                        if (r_cen - 200 < 0) or (r_cen + 200) > 1152 or (c_cen - 200 < 0) or (c_cen + 200) > 1536:
                            pass
                        else:
                            if i == 0:
                                imgrgb1 = image[(r_cen - 200):(r_cen + 200), (c_cen - 200):(c_cen + 200)]
                                if not os.path.isdir(f"{pathstofiles}\\ss_oqc_cropped"):
                                    os.mkdir(f"{pathstofiles}\\ss_oqc_cropped")
                                cv2.imwrite(f"{pathstofiles}\\ss_oqc_cropped\\{os.path.basename(targ)}", imgrgb1)

                            image1 = cv2.imread(str(pred), -1)
                            imgrgb = image1[(r_cen - 200):(r_cen + 200), (c_cen - 200):(c_cen + 200)]
                            if not os.path.isdir(f"{pathstofiles}\\ss_{net}_cropped"):
                                os.mkdir(f"{pathstofiles}\\ss_{net}_cropped")
                            cv2.imwrite(f"{pathstofiles}\\ss_{net}_cropped\\{os.path.basename(pred)}", imgrgb)
                # ax.set_axis_off()
                # plt.tight_layout()
                # plt.show()
                plt.close(fig)
        i = 1

# ss_cropper()
