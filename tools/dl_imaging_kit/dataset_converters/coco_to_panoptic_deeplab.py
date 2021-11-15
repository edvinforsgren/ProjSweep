import sys
import json
import os
from skimage.draw  import polygon2mask
import numpy as np
from skimage.measure import regionprops
from pycocotools import mask as cocomask
from multiprocessing import Pool
import tqdm

import argparse

"""
Script used to work with coco-annotations and convert them into annotations suitable for the deeplab panoptic architecture.
Takes two arguments:
    coco_anno - str path to the coco-annotation file to convert
    target_dir - str that decides the target directory (will then save annotations in that dir under (semantic, offset 
        and centers))
    
Should be noted that iscrowd=1 will be seen as a background class for panoptic purposes.
"""


def extract_instance_mask(coco_anno, img_shape=(704, 520)):
    mask = None

    if coco_anno['iscrowd'] == 0:
        path_list = []
        polygon_path = coco_anno['segmentation'][0]

        for i in range(0, len(polygon_path), 2):
            x = polygon_path[i]
            y = polygon_path[i + 1]
            path_list.append([x, y])

        mask = polygon2mask(img_shape, path_list)

    else:
        m = cocomask.frPyObjects(coco_anno['segmentation'], img_shape[0], img_shape[1])
        mask = cocomask.decode(m)

    return mask


def coco2masks(coco_annotations, img_shape=(704, 520)):
    instances = []
    background = []
    for anno in coco_annotations:
        mask = extract_instance_mask(anno, img_shape)
        if anno['iscrowd'] == 0:
            instances.append(mask)
        else:
            background.append(mask)

    return instances, background


def merge_masks(masks):
    m = np.zeros_like(masks[0])

    for mask in masks:
        m = np.logical_or(m, mask)

    return m

def mask2center(instance_mask):
    M = regionprops(instance_mask.astype(np.uint8))
    return M[0].centroid


def masks2centers(instance_masks):
    centers = []
    for mask in instance_masks:
        centers.append(mask2center(mask))
    return np.array(centers)


def centers2mask(centers, shape=(704, 520)):
    x = centers[:, 0].round().astype(np.int)
    y = centers[:, 1].round().astype(np.int)
    mask = np.zeros(shape)
    mask[x, y] = 1

    return mask


def mask2offset(mask, center):
    offset_output = np.zeros(mask.shape + (2,))
    instance_idx = np.where(mask == 1)
    instance_pos = np.stack([instance_idx[0], instance_idx[1]], axis=1)

    offset = instance_pos - center
    offset_output[instance_idx[0], instance_idx[1]] = offset
    return offset_output


def masks2offsets(masks, centers):
    offsets = None

    for i in range(len(masks)):
        offset = mask2offset(masks[i], centers[i])
        if offsets is not None:
            offsets += offset
        else:
            offsets = offset
    return offsets


def coco2panoptic_deep(coco_annotations, image_shape=(704, 520)):
    instances, backgrounds = coco2masks(coco_annotations, image_shape)
    if len(backgrounds) > 0:
        background = merge_masks(backgrounds)
    else:
        background = np.zeros_like(instances[0])

    centers = masks2centers(instances)
    offsets = masks2offsets(instances, centers)
    center_masks = centers2mask(centers, image_shape)

    instance_mask = merge_masks(instances) * 2
    sem_mask = np.maximum(background, instance_mask)

    return sem_mask, center_masks, offsets


def process_and_save(img_anno):
    semantic_save = save_path + "/semantic/"
    center_save = save_path + "/centers/"
    offset_save = save_path + "/offsets/"

    annotations = []
    for anno in coco_annotations['annotations']:
        if anno['image_id'] == img_anno['id']:
            annotations.append(anno)

    if img_shape is not None:
        sem_seg, centers, offsets = coco2panoptic_deep(annotations)
    else:
        sem_seg, centers, offsets = coco2panoptic_deep(annotations)
    file_name = img_anno['file_name']
    file_name = file_name.split(".")[0]

    np.save(semantic_save + file_name, sem_seg.astype(np.uint8))  # int64 -> uint8
    np.save(center_save + file_name, centers.round().astype(np.uint8))  # float64 -> uint16
    np.save(offset_save + file_name, offsets.round().astype(np.int16))  # float64 -> int16


if __name__ == "__main__":

    global coco_annotations, save_path
    parser = argparse.ArgumentParser(description="Transforms coco annotation to deeplab panoptic")
    parser.add_argument('coco_anno', type=str, help='path to the coco annotation json-file')
    parser.add_argument('target_dir', type=str, help='path to target directory for the annotations')

    args = parser.parse_args()
    json_path = args.coco_anno
    save_path = args.target_dir

    f = open(json_path)
    coco_annotations = json.load(f)

    semantic_save = save_path + "/semantic/"
    center_save = save_path + "/centers/"
    offset_save = save_path + "/offsets/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(semantic_save):
        os.makedirs(semantic_save)

    if not os.path.exists(center_save):
        os.makedirs(center_save)

    if not os.path.exists(offset_save):
        os.makedirs(offset_save)

    image_len = len(coco_annotations['images'])

    def initializer(coco_anno, save_p):
        global coco_annotations, save_path, img_shape
        save_path = save_p
        coco_annotations = coco_anno


    pool = Pool(16, initializer, initargs=(coco_annotations, save_path))

    image_annotations = [img_anno for img_anno in coco_annotations['images']]

    for _ in tqdm.tqdm(pool.imap_unordered(process_and_save, image_annotations), total=len(image_annotations)):
        pass
    print("\n Done!")