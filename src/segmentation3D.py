import os
import pandas as pd
import numpy as np
import sys
from utils.preprocessing.preprocess_utils import load_CT
from utils.seg_measure import last_step
from tqdm import tqdm
from symptom import *
from utils.manifest import *
import csv
from boundary_internal import *

def to_string(dummy_list, sep = ','):
    temp = []
    if type(dummy_list) == list or type(dummy_list) == tuple or type(dummy_list) == np.ndarray:
        for item in dummy_list:
            temp.append('[' + to_string(item)+ ']')
    else:
        temp.append(str(dummy_list))
    return  sep.join(temp)

def to_mm(coord, spacing, origin):
    cd = np.array(coord)
    spc = np.array(spacing)
    ori = np.array(origin)
    assert cd.shape == spc.shape and cd.shape == ori.shape, 'dimensions must be equal!\n'
    return cd * spc + ori

def postprocess(source_path, symptom_classifier_path):
    nodule_list = []
    path_list = []

    for file_name in os.listdir(source_path):
        if 'classified.csv' in file_name:
            nodules = pd.read_csv(os.path.join(source_path, file_name))
            nodule_list = nodule_list + nodules[[u'uid', u'path', u'coordX', u'coordY', u'coordZ', u'diameter', u'det_probability', u'cls_probability']].values.tolist()

    nodule_list = np.array(nodule_list)
    #print 'nodule_list:', nodule_list 
    path_list = list(set([nodule[1].encode('utf-8') for nodule in nodule_list]))
    #print path_list
    #print '*' * 50
    features = []
    index = 1
    symptom_classifier = get_symptom_classifier(symptom_classifier_path)
    bi_classifier = None
    total = len(path_list)
    result = []
    for i, path in enumerate(path_list): 

        result.append([path])
        
        img, spacing, ori, uid = load_CT(path)

        detections = nodule_list[nodule_list[:,0] == uid.decode('utf-8')]

        for detection in detections:
            #print type(detection[2])
            nodule_found = dict()
            centroid = np.array([detection[4].astype(np.float).astype(np.int),\
                                 detection[3].astype(np.float).astype(np.int),\
                                 detection[2].astype(np.float).astype(np.int)])

#             try:

            feature = last_step(img, centroid, spacing, ori)
            symptom = symptom_classification(symptom_classifier, img, centroid, detection[5].astype(np.float), spacing)[0]
            bi_feature = bi_classification(bi_classifier, img, centroid, detection[5].astype(np.float), spacing)

            if feature:

                nodule_found['cls_probability'] = detection[-1].astype(np.float)
                nodule_found['coordX'] = detection[2].astype(np.float)
                nodule_found['coordY'] = detection[3].astype(np.float)
                nodule_found['coordZ'] = detection[4].astype(np.float)
                nodule_found['density'] = feature[-1]
                nodule_found['det_probability'] = detection[-2].astype(np.float)
                nodule_found['greater_diameter_mm'] = feature[0]
                nodule_found['probability'] = (nodule_found['cls_probability'] + nodule_found['det_probability']) / 2.0
                nodule_found['seriesuid'] = uid
                nodule_found['smaller_diameter_mm'] = feature[1]
                nodule_found['probability_ggo'] = symptom[1]
                nodule_found['probability_solid'] = symptom[0]
                nodule_found['probability_mix'] = symptom[2]
                nodule_found['probability_boundary_fenye'] = bi_feature[0]
                nodule_found['probability_boundary_smooth'] = bi_feature[1]
                nodule_found['probability_boundary_maoci'] = bi_feature[2]
                nodule_found['probability_boundary_blur'] = bi_feature[3]
                nodule_found['probability_internal_calcification'] = bi_feature[4]
                result[i].append(nodule_found)
#             except Exception, err:
#                 print err
        print_percentage('Segmentation3D', i + 1, total)
    if len(result) == 1:
        return result[0]
    else:
        return result

#         df.to_csv(os.path.join(target_path, '%s_result.csv'%data_name), index = False)