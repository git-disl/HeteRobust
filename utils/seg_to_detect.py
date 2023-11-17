from gluoncv.data.transforms.presets.segmentation import test_transform
from matplotlib import pyplot as plt
import cv2
from gluoncv.utils.viz import get_color_pallete
import numpy as np

# map segmentation to object detection
def seg_to_detect(
    seg_output_prob,
    class_id_map,
    classes,
    minccsize=200, 
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
):
    # minccsize = 200 # discard those too small
    seg_bboxes_list = []
    seg_ids_list = []
    seg_scores_list = []
    
    seg_predict = np.argmax(seg_output_prob, axis=0)
    #print(seg_predict.shape)

    for class_id in np.unique(seg_predict):
        if class_id == 0: # ignore the background
            continue
        class_img = seg_predict.astype(np.uint8)
        class_img[np.where(seg_predict!=class_id)] = 0
        _, cc_labels = cv2.connectedComponents(class_img)

        # for each connected component for class_id
        for cc_l in np.unique(cc_labels):
            if cc_l == 0: # ignore the background
                continue

            # Filtering <= minsize
            cc_indices = np.argwhere(cc_labels == cc_l)
            if len(cc_indices) < minccsize:
                continue

            xmin = np.min(cc_indices[..., 1])
            ymin = np.min(cc_indices[..., 0])
            xmax = np.max(cc_indices[..., 1])
            ymax = np.max(cc_indices[..., 0])

            obj_prob = seg_output_prob[class_id]
            score = np.mean(obj_prob[cc_labels==cc_l])
            class_unified_id = class_id_map[classes[class_id]]
            #print(cc_l, score)

            seg_bboxes_list.append([xmin, ymin, xmax, ymax])
            seg_ids_list.append(class_unified_id)
            seg_scores_list.append(score)

    seg_bboxes_list = np.array(seg_bboxes_list)
    seg_ids_list = np.array(seg_ids_list)
    seg_scores_list = np.array(seg_scores_list)
    return seg_ids_list, seg_scores_list, seg_bboxes_list
