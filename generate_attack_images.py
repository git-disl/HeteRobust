from dataset_utils.preprocessing import letterbox_image_padded, reverse_letterbox_image_padded
from misc_utils.visualization import visualize_detections
from keras import backend as K
from models.ssd import SSD512
from PIL import Image
from tog.attacks import *
import os
K.clear_session()

from tqdm import tqdm
import time
import random
import pickle


weights = 'model_weights/SSD512.h5'  # TODO: Change this path to the victim model's weights
detector = SSD512(weights=weights)

eps = 64 / 255.       # Hyperparameter: epsilon in L-inf norm
eps_iter = 2 / 255.  # Hyperparameter: attack learning rate
n_iter = 10          # Hyperparameter: number of attack iterations


dataset = 'voc'
benign_basedir = './data/benign'
attack_basedir = './data/attacks/vanishing'


image_list = pickle.load(open(os.path.join(attack_basedir, "image_list.pkl"), "rb"))
print(len(image_list))


with tqdm(total=len(image_list)) as pbar:
    start = time.time()
    for i in image_list:
        benign_img_name = os.path.join(benign_basedir, "voc_2007/"+'{0:04d}'.format(i)+".png")
        with Image.open(benign_img_name) as input_img:
            x_query, x_meta = letterbox_image_padded(input_img, size=detector.model_img_size)
            # mislabel
            x_adv_mislabeling_ll = tog_mislabeling(victim=detector, x_query=x_query, target='ll', n_iter=n_iter, eps=eps, eps_iter=eps_iter)
            x_adv_mislabeling_ll_img = reverse_letterbox_image_padded(x_adv_mislabeling_ll, x_meta)
            x_adv_mislabeling_ll_img.save(os.path.join(attack_basedir, "voc_2007/"+'{0:04d}'.format(i)+".png"))
            # vanishing
            #x_adv_vanishing = tog_vanishing(victim=detector, x_query=x_query, n_iter=n_iter, eps=eps, eps_iter=eps_iter)
            #x_adv_vanishing_img = reverse_letterbox_image_padded(x_adv_vanishing, x_meta)
            #x_adv_vanishing_img.save(os.path.join(attack_basedir, "voc_2007/"+'{0:04d}'.format(i)+".png"))
        pbar.update(1)
    end = time.time()
