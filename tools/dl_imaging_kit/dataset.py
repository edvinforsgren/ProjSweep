from torch.utils.data import Dataset

import cv2
import torch
import numpy as np
import os

class ImagepairDataset(Dataset):
    """A custom torch datasets that loads an image where one can performes two different transforms on it to create a
     input and target version. Suitable for superresolution, deblurring, and similar tasks """

    def __init__(self,
                 dataset,
                 load_func=None,
                 input_transform=None,
                 target_transform=None):
        """

        :param dataset: a list with paths to the images
        :param load_func: will use cv2.imread with -1 flag if this if a load func is not defined.
        :param input_transform: a transform made on the image that will be fed to the model, default a torch tensor
        with channels first.
        :param target_transform: a transform made to the target image, default a torch tensor with channels first.
        """

        self.dataset = dataset
        self.load_func = load_func
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = self.dataset[idx]

        if self.load_func is not None:
            image = self.load_func(path)
        else:
            #Note that cv2 uses BGR as default
            image = cv2.imread(path, -1)

        assert image is not None, f" Could not load {path}"

        if self.input_transform:
            input_img = self.input_transform(image)
        else:
            if len(image.shape) > 2:
                input_img = np.rollaxis(image, 2, 0)
            else:
                input_img = image
            input_img = torch.tensor(input_img)

        if self.target_transform:
            target_img = self.target_transform(image)
        else:
            if len(image.shape) > 2:
                target_img = np.rollaxis(image, 2, 0)
            else:
                target_img = image
            target_img = torch.tensor(target_img)

        return input_img, target_img


class PanopticDeepDataset(Dataset):

    def __init__(self,
                 img_path,
                 target_path,
                 transforms=None,
                 load_func=None):

        self.root = img_path
        self.dataset = [f for f in os.listdir('.') if os.path.isfile(f)]

        if isinstance(target_path, list):
            self.sem_path = target_path[0]
            self.center_path = target_path[1]
            self.offset_path = target_path[2]
        else:
            self.sem_path = target_path + "/semantic/"
            self.center_path = target_path + "/centers/"
            self.offset_path = target_path + "/offsets/"

        self.transforms=transforms
        self.load_func = load_func


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.dataset[idx]

        img_path = self.root + "/" + filename

        if self.load_func is not None:
            image = self.load_func(img_path)
        else:
            #Note that cv2 uses BGR as default
            image = cv2.imread(img_path, -1)

        assert image is not None, f" Could not load {img_path}"

        target_name = filename.replace(".tif", ".npz").replace(".png", ".npz")

        sem_seg = np.load(self.sem_path + target_name)
        centers = np.load(self.center_path + target_name)
        offsets = np.load(self.offset_path + target_name)

        return image, [sem_seg, centers, offsets]

