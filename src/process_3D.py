#!/usr/bin/python
#coding:utf-8

"""
This is the main function for 3d prediction of nodules
"""

import os
import pandas as pd
from detection3D import detection
from classification3D import classification
from utils.utils_3d import *
from utils.manifest import *
import time
#from pspnet import get_model


def process_3d(source_path, target_path, detector_model_path, classifier_model_path, prob_thre = 0.3, iou_thre = False):
    
    """
    this function takes in a list of paths of images (after preprocessing) and directory 
    where all the preprocessed files are stored in and predicts locations of nodules in the image(s)

    @param detector_weights_path: weights of detector model
    @param classifier_weights_path: weights of classifier model
    @param paths: a list of seriesPath(uid), which is used when in classification where paths = glob(seriesPath)[0]
    @param preprocess_dir: directory where all the intermedia preprocessed files and pbb output files are stored
    @param prob_thre: probability threshold used in first nms on detection results
    @param iou_thre: threshold that is used in first nms on detection result
    @param finalfile: csv file that stores the final result after classification

    writes a csv file containing final selected bboxes to finalfile 
    returns a dataframe containing the result before writing to csv
    """
    
    
    print_title('3D detection')
    
    detection(source_path, target_path, detector_model_path, prob_thre, iou_thre)
    
    print_tip('3D detection finished!')    
    
    print_title('3D classification')

    classification(target_path, classifier_model_path)

    print_tip('3D classification finished!')



