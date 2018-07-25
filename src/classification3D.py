#!/usr/bin/python
#coding:utf-8

"""
This file defines classification function, which calls nodule_classification function that classifies
the region of 48x48x48 around each detected box to see if it is a true positive
output returned to executePredict function as the final result of prediction
"""

from utils.utils_3d import process_image
from utils.dbscan import dbscan
import shutil
import gzip
import pandas as pd
import os
import tensorflow as tf
import math
import numpy as np
from glob import glob
from wrnet import get_model
import keras.backend as K
from keras.models import load_model
from utils.manifest import *


def classification(target_path, model_weights_path):
    """
    this function classifies one uid at a time(seriesPath) of resulting filtered detected boxes that possibly contains nodules
    it takes in preprocess_dir and detected results, and draws a 48x48x48 region around each candidate,
    it returns resulting cls_prob
    
    @param classifier: classifier loaded from classification_weights_path from main function
    @param seriesPath: one image uid
    @param preprocess_dir: folder where all the previous files are stored, 
                           files in this folder will be deleted when finished

    returns a list of dictionaries of original entries + cls_probability for each detected box
    """
    
    K.clear_session()
    classifier = get_model()
    classifier.load_weights(model_weights_path)
    
    
    
    OUTPUT_DIM_1 = [1, 48, 48, 48] # dimension of drawn region around each detected box

    npy2csv_output_file = os.path.join(target_path, 'pred3D.csv')
    detector_candidates = pd.read_csv(npy2csv_output_file)
    paths = list(set(detector_candidates[u'path'].values))
    #print paths
    #paths = glob(seriesPath)[0]
    result = []
    total = len(paths)
    for i, path in enumerate(paths):
        (image, spacing, test_x, seriesuid, x_coord, y_coord, z_coord, diameter, detector_prob) = process_image(path, detector_candidates, OUTPUT_DIM_1)

###########################################################################################
        
    
        res = classifier.predict(test_x)
        
        
###########################################################################################
        
        zipped = np.column_stack([seriesuid, [path]*res.shape[0], z_coord, y_coord, x_coord, diameter, detector_prob, res[:,1]]).tolist()
        zipped = [z for z in zipped if float(z[-1]) > 0.05]
        result = result + zipped

        print_percentage('Classification3D', i + 1, total)
    print_gap()

    if len(result):
        rs = np.array(result)
        #print rs.shape
        all_cands = pd.DataFrame(rs, columns = ['uid', 'path', 'coordZ', 'coordY', 'coordX', 'diameter', 'det_probability', 'cls_probability'])
    else:
        all_cands = pd.DataFrame(columns = ['uid', 'path', 'coordZ', 'coordY', 'coordX', 'diameter', 'det_probability', 'cls_probability'])
    all_cands.to_csv(os.path.join(target_path, 'pred3D_c.csv'), index = False)
    
    dbscan(os.path.join(target_path, 'pred3D_c.csv'),os.path.join(target_path, 'pred3D_classified.csv'))
    #return output

