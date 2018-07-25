#!/usr/bin/python
#coding:utf-8

"""
this file defines helper functions used across the prediction process

"""
import os
from glob import glob
from utils.preprocessing.preprocess_utils import load_CT
import pandas as pd
import math
import numpy as np
from numpy import mat, sqrt
from tqdm import tqdm


# this function may not be used in 3d prediction
def parse_name(path):
    """
    this function parses input path to get image uid
    different data source has different imageName parse method

    @param path: path of image
    returns image uid
    """
    if 0:
        imageName = os.path.split(path)[-1].replace(".mhd", "")
        return imageName
    else:
        imageName = os.path.split(path)[0]
        #imageName = os.path.split(imageName)[0]
        imageName = os.path.split(imageName)[-1]
    return imageName

def get_pbb_namelist(dir):
    """
    this function takes in pbb output path (.npy) and 
    convert to np array

    @param dir: pbb_output_path
    returns np array
    """
    samples = glob(os.path.join(dir, "*_pbb.npy"))
    samples = [os.path.split(sample)[-1].replace("_pbb.npy", "") for sample in samples]
    return np.asarray(samples)

# used in converting coordinates
def logit2prob(logit):
    """
    this function takes in logit and convert to probability = exp(x)/(1+exp(x))
    @param logit: x
    returns probability
    """
    odds = np.exp(logit)
    prob = odds / (1 + odds)
    return prob

def distance(x1, x2):
    """
    this function takes in two vectors x1 and x2 of same size and calculates their sqrt distance
    """
    assert len(x1) == 3
    assert len(x2) == 3
    vector1 = mat(x1)
    vector2 = mat(x2)
    #print vector1.shape
    #print vector2.shape
    return sqrt(np.dot((vector1 - vector2),(vector1-vector2).T).astype(np.float)[0][0])


def nms_inner(candidates, ori_indexes=None):
    '''candidates: a numpy array 
    candidates of a SINGLE ct!'''
    candidates = candidates[-candidates[:, 0].argsort()]
    if len(candidates) > 500: 
        candidates = candidates[:1000]
    keep_bboxes = np.ones(len(candidates), dtype=np.bool)
    for i in range(len(candidates)):
        if keep_bboxes[i]:
            min_distance = candidates[i][4]/2
            # Computer overlap with bboxes which are following
            for j in np.arange(i+1,len(candidates)):
                if keep_bboxes[j]:
                    not_overlap = distance(candidates[i][1:4], candidates[j][1:4]) > min_distance
                    keep_bboxes[j] = not_overlap
            # Overlap threshold for keeping + checking part of the same class
    idxes = np.where(keep_bboxes)
    # if np.any(ori_indexes==None):
    idxes = [i for i in idxes[0]]
    return idxes
    # else:
        # return ori_indexes[idxes]


def nms(can, prob_thre = 0.5, iou_thre = False):
    """
    this function uses non-maximum supression(nms) method to filter candidates
    it first reads in csv results stored in npy2csv_output_file after detection step
    then it drops candidates with probabilities less than prob_thre
    it generates a new csv file under preprocess_dir named preprocess_dir + "detector.csv"

    @param can: dataframe that stores results from detection, in the format of 
                               [seriesid, x, y, z, d, prob]
    @param preprocess_dir: directory where this function writes nms results to 
    @param prob_thre: probability threshold below which candidates are dropped
    @param iou_thre: iou threshold above which boxes are dropped

    writes one csv file to preprocess_dir + 'detector.csv' file
    returns none

    """
    can = can[can["probability"] > prob_thre]
    name_list = list(can.seriesuid.drop_duplicates())
    keep = []
    # delete = []
    # flag = 0
    for sample in name_list:
        #print "processing nms: ", sample
        # temp = can[can.seriesuid == sample].sort_values(["probability"], ascending=False)
        candidates = can[can.seriesuid == sample].sort_values(["probability"], ascending=False)
        candidates = candidates.values
        if len(candidates)>1000:
            candidates = candidates[:500]
        kept_idxs = nms_inner(candidates)
        keep.extend(kept_idxs)
    #     index = temp.index
    #     if len(temp > 0):
    #         flag += 1
    #         # while len(keep) + len(delete) < len(temp):
    #         while len(index) > 0:
    #             keep.append(index[0])
    #             a = [temp.loc[index[0]].coordX, temp.loc[index[0]].coordY, temp.loc[index[0]].coordZ]
    #             if iou_thre = False:
    #                 min_distance = temp.loc[index[0]].diameter_mm / 2
    #             else:
    #                 min_distance = iou_thre
    #             index = index.drop(index[0])
    #             if len(index) > 0:
    #                 for ix in index:
    #                     b = [temp.loc[ix].coordX, temp.loc[ix].coordY, temp.loc[ix].coordZ]
    #                     if distance(a, b) < min_distance:
    #                         index = index.drop(ix)
    #                         delete.append(ix)    
    #         print "keep: ", len(keep)
    #         print "delete: ", len(delete)
    can_nms = can.loc[keep]
    return can_nms


def process_image(image_path, det_candidates, OUTPUT_DIM_1):
    """
    this function cuts patches of [1, 48, 48, 48] around each detected bounding box used in classification

    @param image_path: full path of an image
    @param det_candidates: dataframe of detected candidates after filtering, must contain key seriesuid
    @param OUTPUT_DIM_1: dimension of drawn region around each detected box = [1, 48, 48, 48]

    returns image, spacing, test_x, seriesuid, x_coord, y_coord, z_coord, diameter, detector_prob
    """
    #image_name = str(image_path.split('/')[-1]) # parse for uid
    image, spacing, origin, image_name = load_CT(image_path)
    #print('image', image_path, 'loaded')
    #print image_path.split('/')
    #print image_name
    #print candidates.iloc[1].seriesuid
    offset_1 = [OUTPUT_DIM_1[1]/2, OUTPUT_DIM_1[2]/2, OUTPUT_DIM_1[3]/2]
    indices = det_candidates[det_candidates['uid'] == image_name].index
    z_len = image.shape[0]
    y_len = image.shape[1]
    x_len = image.shape[2]
    #print "series number %s has shape of %d*%d*%d" %(image_name, z_len, y_len, x_len)
    #print det_candidates.iloc[0]

    test_x = []
    seriesuid = []
    z_coord = []
    y_coord = []
    x_coord = []
    diameter = []
    detector_prob = []
    # loop through the candidates within this image
    for n, i in enumerate(indices):
        # get row data and nodule voxel coords
        # one bbox
        row = det_candidates.iloc[i]
        z = int(row.coordZ)
        y = int(row.coordY)
        x = int(row.coordX)
        # the patch that is around this bounding box
        zyx_1 = np.zeros(OUTPUT_DIM_1, dtype=np.int16) - 1000       
        try:
            for z2 in range(OUTPUT_DIM_1[1]):
                for y2 in range(OUTPUT_DIM_1[2]):
                    for x2 in range(OUTPUT_DIM_1[3]):
                        if (z-offset_1[0]+z2 >= z_len or y-offset_1[1]+y2 >= y_len or x-offset_1[2]+x2 >= x_len):
                            zyx_1[0, z2, y2, x2] = 0
                        else:    
                            zyx_1[0, z2, y2, x2] = image[z-offset_1[0]+z2, y - offset_1[1] + y2, x - offset_1[2] + x2]
        except Exception, e:
            print 'str(Exception):\t', str(Exception)
            print 'str(e):\t\t', str(e)
            continue
        # one patch ready, append to the list 
        test_x.append(zyx_1) 
        seriesuid.append(image_name)
        # append coordinates of original x, y, z of bounding boxes
        x_coord.append(row.coordX)
        y_coord.append(row.coordY)
        z_coord.append(row.coordZ)
        diameter.append(row.diameter)
        detector_prob.append(np.array(row.probability))
    test_x = np.array(test_x)
    # switch axes
    test_x = np.swapaxes(test_x, 1, 2)
    test_x = np.swapaxes(test_x, 2, 3)
    test_x = np.swapaxes(test_x, 3, 4)
    # normalize
    test_x = (test_x - test_x.mean())/test_x.std()
    return image, spacing, test_x, np.array(seriesuid), np.array(x_coord), np.array(y_coord), np.array(z_coord), np.array(diameter), np.array(detector_prob)

        

