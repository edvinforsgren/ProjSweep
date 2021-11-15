"""
Created on 2020-07-10. 12:39 

@authors: Christoffer Edlund and Edvin Forsgren
"""
from torch.utils.data import Dataset
import torch
import cv2
import numpy as np
import random
import os
import torchvision.transforms as tf
from PIL import Image
from torch.utils.data.dataloader import DataLoader
import datetime
import json


class ImagepairDataset(Dataset):
    """A custom torch datasets that loads an image where one can performes two different transforms on it to create a
     input and target version. Suitable for superresolution, deblurring, and similar tasks """

    def __init__(self,
                 dataset,
                 load_func=None,
                 input_transform=None,
                 target_transform=None,
                 train=True,
                 get_info=None):
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
        self.train = train
        self.get_info = get_info

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.train:
            if torch.is_tensor(idx):
                idx = idx.tolist()

            path = self.dataset[idx]

            if self.load_func is not None:
                image = self.load_func(path)
                assert image is not None, f"Could not find path: {path}"
            else:
                # Note that cv2 uses BGR as default
                image = cv2.imread(path, -1)

            assert image is not None, f" Could not load {path}"
            np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)
            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)

            if self.input_transform:
                input_img = self.input_transform(image)
            else:
                if len(image.shape) > 2:
                    input_img = np.rollaxis(image, 2, 0)
                else:
                    input_img = image
                input_img = torch.tensor(input_img)
            random.seed(seed)
            torch.manual_seed(seed)
            if self.target_transform:
                target_img = self.target_transform(image)
            else:
                if len(image.shape) > 2:
                    target_img = np.rollaxis(image, 2, 0)
                else:
                    target_img = image
                target_img = torch.tensor(target_img)

            return input_img, target_img
        elif not self.train:
            if torch.is_tensor(idx):
                idx = idx.tolist()

            path = self.dataset[idx]

            if self.load_func is not None:
                image = self.load_func(path)
                assert image is not None, f"Could not find path: {path}"
            else:
                # Note that cv2 uses BGR as default
                image = cv2.imread(path, -1)
            input_scale, target_scale = 255 / np.max(image[0]), 255 / np.max(image[1])
            input_mean, target_mean = np.mean(image[0]), np.mean(image[1])
            input_data = {"Scale": input_scale.astype('float'),
                          "Mean": input_mean.astype('float')}
            target_data = {"Scale": target_scale.astype('float'),
                           "Mean": target_mean.astype('float')}
            assert image is not None, f" Could not load {path}"
            np.random.seed(idx)
            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)

            if self.input_transform:
                input_img = self.input_transform(image)
            else:
                if len(image.shape) > 2:
                    input_img = np.rollaxis(image, 2, 0)
                else:
                    input_img = image
                input_img = torch.tensor(input_img)
            random.seed(seed)
            torch.manual_seed(seed)
            if self.target_transform:
                target_img = self.target_transform(image)
            else:
                if len(image.shape) > 2:
                    target_img = np.rollaxis(image, 2, 0)
                else:
                    target_img = image
                target_img = torch.tensor(target_img)
            if self.get_info:
                input_img = (input_img, json.dumps(input_data))
                target_img = (target_img, json.dumps(target_data))
            return input_img, target_img


def create_dataloader(path_a, path_b, split=None, batch_size=2, num_workers=8, crop_size=(256, 256), random_crop=True,
                      train=True, on_gpu=True,
                      noise=False, gauss=False, get_info=None, repeat=1):
    def to_tensor(img):
        img = np.clip(img, a_min=1, a_max=np.max(img))  # To avoid np.log(0) error. Generally, the images min is > 1
        img = np.log(img)
        img = torch.tensor(np.array(img)).float()
        img = img.unsqueeze(0)
        return img

    def data_load_func(path):
        path_a, path_b = path
        img_a = cv2.imread(path_a, -1)
        img_b = cv2.imread(path_b, -1)

        assert img_a is not None, f"Could not find {path_a}"
        assert img_b is not None, f"Could not find {path_b}"

        return img_a, img_b

    def transform_a(image):
        image, _ = image
        image = aug_a(Image.fromarray(image))
        return image

    def transform_b(image):
        _, image = image
        image = aug_b(Image.fromarray(image))
        return image

    if random_crop:
        aug_list = [tf.RandomCrop(crop_size),
                    tf.RandomHorizontalFlip(0.25),
                    tf.RandomVerticalFlip(0.25),
                    to_tensor]
    else:
        aug_list = [tf.CenterCrop(crop_size),
                    tf.RandomHorizontalFlip(0.25),
                    tf.RandomVerticalFlip(0.25),
                    to_tensor]

    aug_b = tf.Compose(aug_list)
    aug_list_a = aug_list.copy()
    aug_a = tf.Compose(aug_list_a)

    paths_a = [path_a + "/" + f for f in os.listdir(path_a) if os.path.isfile(path_a + "/" + f)]
    paths_b = [path_b + "/" + f for f in os.listdir(path_b) if os.path.isfile(path_b + "/" + f)]

    if repeat == 1:
        paths_b.extend(paths_b)
        paths_a.extend(paths_a)
        paths_b.extend(paths_b)
        paths_a.extend(paths_a)

    paths_a.sort()
    paths_b.sort()
    move_paths = ["A11", "B10", "C6", "D4", "F8", "G3", "H2", "E7", "A2",
                  "B3"]  # To get different wells in training and validation
    moved_paths = []
    for path in paths_a:
        for match in move_paths:
            if path.__contains__(match):
                moved_paths.append(path)
    for path in moved_paths:
        paths_a.remove(path)
    paths_a.extend(moved_paths)
    moved_paths = []

    for path in paths_b:
        for match in move_paths:
            if path.__contains__(match):
                moved_paths.append(path)
    for path in moved_paths:
        paths_b.remove(path)
    paths_b.extend(moved_paths)

    paths = list(zip(paths_a, paths_b))
    if split is not None:
        split = int(split * len(paths))
        if train:
            paths = paths[:split]
        else:
            paths = paths[split:]

    dataset = ImagepairDataset(paths, load_func=data_load_func, input_transform=transform_a,
                               target_transform=transform_b, train=train, get_info=get_info)
    if on_gpu:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=train, pin_memory=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=train)
    return loader
