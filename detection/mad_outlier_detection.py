#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-28 16:27:19
# @Author  : Bolun Wang (bolunwang@cs.ucsb.edu)
# @Link    : http://cs.ucsb.edu/~bolunwang

import os
import sys
import time

import numpy as np
from tensorflow.keras.preprocessing import image


##############################
#        PARAMETERS          #
##############################

RESULT_DIR = '0.1_F2F_BadNet_5_ShallowNetV3_trigger_white_100_0'  # directory for storing results
#RESULT_DIR = 'results_clean'  # directory for storing results
IMG_FILENAME_TEMPLATE = 'F2F_visualize_%s_label_%d.png'  # image filename template for visualization results

# input size
IMG_ROWS = 256
IMG_COLS = 256
IMG_COLOR = 3
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_COLOR)

NUM_CLASSES = 6  # total number of classes in the model

##############################
#      END PARAMETERS        #
##############################


def outlier_detection(l1_norm_list, idx_mapping,consistency_ratio):

    consistency_constant = 1.4826  # if normal distribution
    #consistency_constant = 0.75 # if normal distribution
    consistency_constant = consistency_constant *consistency_ratio
    consistency_constant = consistency_constant
    median = np.median(l1_norm_list)
    mad = consistency_constant * np.median(np.abs(l1_norm_list - median))
    min_mad = np.abs(np.min(l1_norm_list) - median) / mad

    print('The detection raio is %s'% consistency_ratio)
    print('median: %f, MAD: %f' % (median, mad))
    print('anomaly index: %f' % min_mad)

    flag_list = []
    for y_label in idx_mapping:
        if l1_norm_list[idx_mapping[y_label]] > median:
            continue
        if np.abs(l1_norm_list[idx_mapping[y_label]] - median) / mad > 2:
            flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]]))

    if len(flag_list) > 0:
        flag_list = sorted(flag_list, key=lambda x: x[1])

    print('flagged label list: %s' %
          ', '.join(['%d: %2f' % (y_label, l_norm)
                     for y_label, l_norm in flag_list]))
  

    pass


def analyze_pattern_norm_dist():

    mask_flatten = []
    idx_mapping = {}

    for y_label in range(NUM_CLASSES):
        mask_filename = IMG_FILENAME_TEMPLATE % ('mask', y_label)
        if os.path.isfile('%s/%s' % (RESULT_DIR, mask_filename)):
            img = image.load_img(
                '%s/%s' % (RESULT_DIR, mask_filename),
                color_mode='grayscale',
                target_size=INPUT_SHAPE)
            mask = image.img_to_array(img)
            mask /= 255
            mask = mask[:, :, 0]

            mask_flatten.append(mask.flatten())

            idx_mapping[y_label] = len(mask_flatten) - 1

    l1_norm_list = [np.sum(np.abs(m)) for m in mask_flatten]

    print('%d labels found' % len(l1_norm_list))

    outlier_detection(l1_norm_list, idx_mapping,1)
    outlier_detection(l1_norm_list, idx_mapping,0.9)
    outlier_detection(l1_norm_list, idx_mapping,0.8)
    outlier_detection(l1_norm_list, idx_mapping,0.7)
    outlier_detection(l1_norm_list, idx_mapping,0.6)
    outlier_detection(l1_norm_list, idx_mapping,0.5)
    pass


if __name__ == '__main__':

    print('%s start' % sys.argv[0])

    start_time = time.time()
    analyze_pattern_norm_dist()
    elapsed_time = time.time() - start_time
    print('elapsed time %.2f s' % elapsed_time)
