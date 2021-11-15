# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/detectron2/blob/master/detectron2/evaluation/coco_evaluation.py
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# Modified by Christoffer Edlund to make eval work with more instances
# ------------------------------------------------------------------------------

import logging
from collections import OrderedDict
import os
import glob
import copy
import json

import numpy as np

from fvcore.common.file_io import PathManager
import pycocotools.mask as mask_util
import types


class COCOInstanceEvaluator:
    """
    Evaluate COCO instance segmentation
    """
    def __init__(self, output_dir=None, train_id_to_eval_id=None,
                 gt_dir='./datasets/coco/annotations/instances_val2017.json'):
        """
        Args:
            output_dir (str): an output directory to dump results.
            train_id_to_eval_id (list): maps training id to evaluation id.
            gt_dir (str): path to ground truth annotations.
        """
        if output_dir is None:
            raise ValueError('Must provide a output directory.')
        self._output_dir = output_dir
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
        self._train_id_to_eval_id = train_id_to_eval_id

        self._predictions = []
        self._predictions_json = os.path.join(output_dir, 'predictions.json')

        self._logger = logging.getLogger(__name__)

        self._gt_dir = gt_dir

    def update(self, instances, image_filename=None):
        if image_filename is None:
            raise ValueError('Need to provide image_filename.')
        num_instances = len(instances)

        for i in range(num_instances):
            pred_class = instances[i]['pred_class']
            if self._train_id_to_eval_id is not None:
                pred_class = self._train_id_to_eval_id[pred_class]
            image_id = int(os.path.basename(image_filename).split('.')[0])
            score = instances[i]['score']
            mask = instances[i]['pred_mask'].astype("uint8")
            # use RLE to encode the masks, because they are too large and takes memory
            # since this evaluator stores outputs of the entire dataset
            mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")

            self._predictions.append(
                {
                    'image_id': image_id,
                    'category_id': pred_class,
                    'segmentation': mask_rle,
                    'score': float(score)
                }
            )

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP" and "AP50".
        """
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        def summarize_2(self):
            '''
            Compute and display summary metrics for evaluation results.
            Note this functin can *only* be applied on the default parameter setting
            '''

            print("In method")

            def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=2000):
                p = self.params
                iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
                titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
                typeStr = '(AP)' if ap == 1 else '(AR)'
                iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                    if iouThr is None else '{:0.2f}'.format(iouThr)

                aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
                mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
                if ap == 1:
                    # dimension of precision: [TxRxKxAxM]
                    s = self.eval['precision']
                    # IoU
                    if iouThr is not None:
                        t = np.where(iouThr == p.iouThrs)[0]
                        s = s[t]
                    s = s[:, :, :, aind, mind]
                else:
                    # dimension of recall: [TxKxAxM]
                    s = self.eval['recall']
                    if iouThr is not None:
                        t = np.where(iouThr == p.iouThrs)[0]
                        s = s[t]
                    s = s[:, :, aind, mind]
                if len(s[s > -1]) == 0:
                    mean_s = -1
                else:
                    mean_s = np.mean(s[s > -1])
                print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
                return mean_s

            def _summarizeDets():

                stats = np.zeros((12,))
                stats[0] = _summarize(1, maxDets=self.params.maxDets[2])
                stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
                stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
                stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
                stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
                stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
                stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
                stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
                stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
                stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
                stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
                stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
                return stats

            def _summarizeKps():
                stats = np.zeros((10,))
                stats[0] = _summarize(1, maxDets=20)
                stats[1] = _summarize(1, maxDets=20, iouThr=.5)
                stats[2] = _summarize(1, maxDets=20, iouThr=.75)
                stats[3] = _summarize(1, maxDets=20, areaRng='medium')
                stats[4] = _summarize(1, maxDets=20, areaRng='large')
                stats[5] = _summarize(0, maxDets=20)
                stats[6] = _summarize(0, maxDets=20, iouThr=.5)
                stats[7] = _summarize(0, maxDets=20, iouThr=.75)
                stats[8] = _summarize(0, maxDets=20, areaRng='medium')
                stats[9] = _summarize(0, maxDets=20, areaRng='large')
                return stats

            if not self.eval:
                raise Exception('Please run accumulate() first')
            iouType = self.params.iouType
            if iouType == 'segm' or iouType == 'bbox':
                summarize = _summarizeDets
            elif iouType == 'keypoints':
                summarize = _summarizeKps
            self.stats = summarize()


        if self._gt_dir is None:
            raise ValueError('Must provide coco gt path for evaluation.')

        self._logger.info("Evaluating results under {} ...".format(self._output_dir))

        coco_gt = COCO(self._gt_dir)

        coco_results = copy.deepcopy(self._predictions)
        # When evaluating mask AP, if the results contain bbox, cocoapi will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in coco_results:
            c.pop("bbox", None)

        with PathManager.open(self._predictions_json, "w") as f:
            f.write(json.dumps(coco_results))

        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
        # coco_eval.params.catIds = [1]
        coco_eval.params.maxDets = [100, 500, 2000]
        # coco_eval.imgIds = [1309499]


        coco_eval.summarize = types.MethodType(summarize_2, coco_eval)

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats