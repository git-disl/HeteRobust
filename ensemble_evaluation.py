import numpy as np
import mxnet as mx
from mxnet import gluon
import os
from tqdm import tqdm
import time
import gluoncv as gcv
from gluoncv.data import VOCDetection
from matplotlib import pyplot as plt
from multiprocessing import cpu_count
CPU_COUNT = cpu_count()

import pickle
import numpy as np
from PIL import Image
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from utils.ensemble_methods import getEnsembleResults

net_names = ['SSD512', 'YOLOv3', 'FasterRCNN', 'DeepLabv3', 'PSPNet']
voc_classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

dataset = 'voc'
benign_basedir = './data/benign'
results_basedir = './data/attacks/vanishing/voc_2007/results'
attack_basedir = './data/attacks/vanishing'
attack_dataset = 'voc_2007'

if dataset == 'voc':
    from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
    val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=voc_classes)
elif dataset == 'coco':
    val_metric = COCODetectionMetric(val_dataset, 'coco_eval', cleanup=True,)

# Load image list and gt labels
image_list = pickle.load(open(os.path.join(attack_basedir, "image_list.pkl"), "rb"))
print(len(image_list))

gt_dict = pickle.load(open(os.path.join(benign_basedir, dataset + "-gt.pkl"), 'rb'))
gt_bboxes = gt_dict['gt_bboxes']
gt_ids = gt_dict['gt_ids']
gt_difficults = gt_dict['gt_difficults']


# Ensemble Study
result_dict_list = []
det_bboxes_list = []
det_ids_list = []
det_scores_list = []
adv_bboxes_list = []
adv_ids_list = []
adv_scores_list = []

for i in range(len(net_names)):
    result_dict = pickle.load(open(os.path.join(results_basedir, net_names[i]), 'rb'))
    result_dict_list.append(result_dict)
    det_bboxes = result_dict['det_bboxes']
    det_bboxes_list.append(det_bboxes)
    det_ids = result_dict['det_ids']
    det_ids_list.append(det_ids)
    det_scores = result_dict['det_scores']
    det_scores_list.append(det_scores)
    
    adv_bboxes = result_dict['adv_bboxes']
    adv_bboxes_list.append(adv_bboxes)
    adv_ids = result_dict['adv_ids']
    adv_ids_list.append(adv_ids)
    adv_scores = result_dict['adv_scores']
    adv_scores_list.append(adv_scores)


# Ensemble Results
val_metric.reset()
with tqdm(total=len(image_list)) as pbar:
    for i in range(len(image_list)):
        ids, scores, bboxes = getEnsembleResults(adv_ids_list, adv_scores_list, adv_bboxes_list, i, [0, 1, 2, 3, 4])
        val_metric.update(bboxes, ids, scores, 
                          gt_bboxes[image_list[i]], 
                          gt_ids[image_list[i]],
                          gt_difficults[image_list[i]])
        pbar.update(1)

names, values = val_metric.get()
for k, v in zip(names, values):
    print(k, v)

