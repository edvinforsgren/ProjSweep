from torch.nn import functional as F
import torch
from collections import OrderedDict
import numpy as np

def find_instance_center(
        center_htmp,
        threshold: float,
        htmp_kernel: int = 7,
        top_k: int = None

    ):
    """
    This funtion performes a NMS by moving a max pooling kernel along the input-heatmap, (instance-htmp), if the
    max-pool yields the same number as the current pixel value, a local maxima have been found. Make sure to set a
    threshold value to suppress maxima with a low value (false positives).

    :param instance_htmp:
    :param threshold:
    :param htmp_kernel:
    :param top_k:
    :return:
    """

    if center_htmp.size(0) != 1:
        raise ValueError("Only supports inference of batch size = 1")

    center_htmp = F.threshold(center_htmp, threshold, -1)
    htmp_pad = (center_htmp - 1) // 2


    pooled_htmp = F.max_pool2d(center_htmp, kernel_size=htmp_kernel, stride=1, padding=htmp_pad)
    center_htmp[center_htmp != pooled_htmp] = -1

    center_htmp = center_htmp.squeeze()
    assert len(center_htmp.size()) == 2

    instance_centers = torch.nonzero(center_htmp > 0)

    if top_k is None:
        return instance_centers

    elif instance_centers.size(0) < top_k:
        return instance_centers

    else:
        top_k_centers, _ = torch.topk(torch.flatten(instance_centers), top_k)
        return torch.nonzero(instance_centers > top_k_centers[-1])


def group_pixels(
        centers,
        offsets
):
    """
    Inputs coordinates of predicted centers and offsets. Groups pixels into instances based on the closest center after
    offsets have been taken into account for each pixel.
    :param centers: Tensor with shape [K, 2], representing top k center positions
    :param offsets: Tensor with shape [N, 2, H, W] representing the offset for each pixel to it's instance center
    :return: Tensor with shape [1, H, W], where each pixel now have it's class agnostic instance id.
    """

    assert offsets.size(0) != 1, "Only batch size equal to one"

    offsets = offsets.squeeze(0)
    height, width = offsets.size()[1:]

    dtype = offsets.dtype
    device = offsets.device

    y_coords = torch.arange(height, dtype=dtype, device=device).repeat(1, width, 1).transpose(1,2)
    x_coords = torch.arange(width, dtype=dtype, device=device).repeat(1, height, 1)
    coords = torch.cat((y_coords, x_coords), dim=0)

    center_locations = coords + offsets
    center_locations = center_locations.reshape(2, height * width).transpose(1, 0)

    # centers: [K, 2] -> [K, 1, 2]
    # center_locations: [H*W, 2] -> [1, H*W, 2]
    centers = centers.unsqueeze(1)
    center_locations = center_locations.unsqueeze(0)

    #Lets measure all the distances! [K, H*W]
    distances = torch.norm(centers - center_locations, dim=-1)

    #And convert each pixel value to an instance value based on distance to center (id=0 is reserved for background)
    instance_map = torch.argmin(distances, dim=0).reshape((1, height, width)) + 1

    return instance_map


def process_instance_segmentation(
        sem_seg,
        center_htmp,
        offsets,
        thing_list,
        threshold: float=0.1,
        nms_kernel: int=3,
        top_k: int=None,
        thing_seg=None
    ):
    """
    Post-processing for instance segmentation, gets class agnostic instance id map.
    Arguments:
        sem_seg: A Tensor of shape [1, H, W], predicted semantic label.
        ctr_hmp: A Tensor of shape [N, 1, H, W] of raw center heatmap output, where N is the batch size,
            for consistent, we only support N=1.
        offsets: A Tensor of shape [N, 2, H, W] of raw offset output, where N is the batch size,
            for consistent, we only support N=1. The order of second dim is (offset_y, offset_x).
        thing_list: A List of thing class id.
        threshold: A Float, threshold applied to center heatmap score.
        nms_kernel: An Integer, NMS max pooling kernel size.
        top_k: An Integer, top k centers to keep.
        thing_seg: A Tensor of shape [1, H, W], predicted foreground mask, if not provided, inference from
            semantic prediction.
    Returns:
        A Tensor of shape [1, H, W] (to be gathered by distributed data parallel).
        A Tensor of shape [1, K, 2] where K is the number of center points. The order of second dim is (y, x).
    """
    if thing_seg is None:
        thing_seg = torch.zeros_like(sem_seg)
        for thing_class in thing_list:
            thing_seg[sem_seg == thing_class] = 1

    centers = find_instance_center(center_htmp, threshold=threshold, nms_kernel=nms_kernel, top_k=top_k)

    # If no instance centers where found
    if centers.size(0) == 0:
        return torch.zeros_like(sem_seg), centers.unsqueeze(0)

    instance_seg = group_pixels(centers, offsets)
    return thing_seg * instance_seg, centers.unsqueeze(0)


def merge_instance_and_semantic(
        sem_seg,
        ins_seg,
        label_divisor:int,
        thing_list,
        stuff_area_threshold:int,
        void_label:int
    ):
    """

    :param sem_seg:
    :param ins_seg:
    :param label_divisor: panoptic id = semantic id * label_divisor + instance_id
    :param thing_list:
    :param stuff_area_threshold:
    :param void_label:
    :return:
    """

    pan_seg = torch.zeros_like(sem_seg) + void_label
    thing_seg = ins_seg > 0
    sem_thing_seg = torch.zeros_like(sem_seg)
    for thing_class in thing_list:
        sem_thing_seg[sem_seg == thing_class] = 1

    # Keeps track of instance id for each class
    class_id_tracker = {}

    # Let's do some majority voting
    instance_ids = torch.unique(ins_seg)
    for ins_ids in instance_ids:
        if ins_ids == 0:
            continue

        thing_mask = (ins_seg == ins_ids) & (sem_thing_seg == 1)
        if torch.nonzero(thing_mask).size(0) == 0:
            continue
        class_id, _ = torch.mode(sem_seg[thing_mask].view(-1,))
        if class_id.item() in class_id_tracker:
            new_ins_id = class_id_tracker[class_id.item()]
        else:
            class_id_tracker[class_id.item()] = 1
            new_ins_id = 1
        class_id_tracker[class_id.item()] += 1
        pan_seg[thing_mask] = class_id * label_divisor + new_ins_id

    # Handle unoccupied area
    class_ids = torch.unique(sem_seg)
    for class_id in class_ids:
        if class_id.item() in thing_list:
            # Found a thing class
            continue
        stuff_mask = (sem_seg == class_id) & (~thing_seg)
        area = torch.nonzero(stuff_mask).size(0)
        if area >= stuff_area_threshold:
            pan_seg[stuff_mask] = class_id * label_divisor
        return pan_seg


def get_semantic_segmentation(sem_seg):
    """
    Post-processing for semantic segmentation branch.
    Arguments:
        sem: A Tensor of shape [N, C, H, W], where N is the batch size, for consistent, we only
            support N=1.
    Returns:
        A Tensor of shape [1, H, W] (to be gathered by distributed data parallel).
    Raises:
        ValueError, if batch size is not 1.
    """
    if sem_seg.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')
    sem_seg = sem_seg.squeeze(0)
    return torch.argmax(sem_seg, dim=0, keepdim=True)


def process_panoptic_segmentation(
        sem_seg,
        center_htmp,
        offsets,
        thing_list,
        label_divisor,
        stuff_area_threshold,
        void_label,
        threshold=0.1,
        nms_kernel=3,
        top_k=None,
        foreground_mask=None
    ):


    if sem_seg.dim() != 3:
        raise ValueError(f"Semantic prediction with un-supported dimentsion: {sem_seg.dim()}")
    if sem_seg.dim() == 4 and sem_seg.size(0) != 1:
        raise ValueError("Only supports a batch size of 1")
    if center_htmp.size(0) != 1 or offsets.size(0) != 1:
        raise ValueError("Only supports a batch size of 1")
    if foreground_mask is not None:
        if foreground_mask.dim() != 4 and foreground_mask.dim() != 3:
            raise ValueError(f"Foreground prediction with un-supported dimension {foreground_mask.dim()}")

    if sem_seg.dim() == 4:
        semantic_seg = get_semantic_segmentation()
        pass
    else:
        semantic_seg = sem_seg

    if foreground_mask is not None:
        if foreground_mask.dim() == 4:
            thing_seg = get_semantic_segmentation()
            pass
        else:
            thing_seg = foreground_mask
    else:
        thing_seg = None

    instance_seg, center = process_instance_segmentation(semantic_seg, center_htmp, offsets, thing_list,
                                                     threshold=threshold, nms_kernel=nms_kernel, top_k=top_k,
                                                     thing_seg=thing_seg)
    panoptic_seg = merge_instance_and_semantic(semantic_seg, instance_seg, label_divisor, thing_list,
                                               stuff_area_threshold, void_label)

    return panoptic_seg


def get_cityscapes_instance_format(panoptic, sem, ctr_hmp, label_divisor, score_type="instance"):
    """
    Get Cityscapes instance segmentation format.
    Arguments:
        panoptic: A Numpy Ndarray of shape [H, W].
        sem: A Numpy Ndarray of shape [C, H, W] of raw semantic output.
        ctr_hmp: A Numpy Ndarray of shape [H, W] of raw center heatmap output.
        label_divisor: An Integer, used to convert panoptic id = semantic id * label_divisor + instance_id.
        score_type: A string, how to calculates confidence scores for instance segmentation.
            - "semantic": average of semantic segmentation confidence within the instance mask.
            - "instance": confidence of heatmap at center point of the instance mask.
            - "both": multiply "semantic" and "instance".
    Returns:
        A List contains instance segmentation in Cityscapes format.
    """
    instances = []

    pan_labels = np.unique(panoptic)
    for pan_lab in pan_labels:
        if pan_lab % label_divisor == 0:
            # This is either stuff or ignored region.
            continue

        ins = OrderedDict()

        train_class_id = pan_lab // label_divisor
        ins['pred_class'] = train_class_id

        mask = panoptic == pan_lab
        ins['pred_mask'] = np.array(mask, dtype='uint8')

        sem_scores = sem[train_class_id, ...]
        ins_score = np.mean(sem_scores[mask])
        # mask center point
        mask_index = np.where(panoptic == pan_lab)
        center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
        ctr_score = ctr_hmp[int(center_y), int(center_x)]

        if score_type == "semantic":
            ins['score'] = ins_score
        elif score_type == "instance":
            ins['score'] = ctr_score
        elif score_type == "both":
            ins['score'] = ins_score * ctr_score
        else:
            raise ValueError("Unknown confidence score type: {}".format(score_type))

        instances.append(ins)

    return instances