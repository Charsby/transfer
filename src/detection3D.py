#!/usr/bin/python
#coding:utf-8

"""
This file defines detection function that takes in preprocessed files and outputs detected pbbs
and write to csv file that contains coordinates of all detected boxes w/ probabilities

"""
#from nodule_serving_test import nodule_detection_serving
from utils.utils_3d import logit2prob, get_pbb_namelist, nms_inner
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import itertools
from detector3D import predict_3D
from utils.dbscan import *

def detection(source_path, target_path, model_weights_path, prob_thre = 0.3, iou_thre = False):
    """
    this function takes in the directory where all the preprocessed files are stored and 
    does detection of nodules using the following steps:

    it first imports model weights and output detected pbbs to pbb_output_path(same as preprocess_dir),
    then it converts each pbb back to its original coordinates using logit2prob and filter using nms and saves in a
    csv file.

    @param detector_weights_path: path containing the weights of model
    @param preprocess_dir: folder where all the cleaned .npy file and the extended boxes are stored
    @param prob_thre: prob_thre used to filter detected bounding boxes
    @param iou_thre: iou threshold used to filter detected bounding boxes

    writes one csv file in npy2csv_output_file that is the coordinates and prob of detected nodules
    returns none
    """
   
    pbb_dir = os.path.join(target_path, 'pbb')
    path_uid_csv = os.path.join(source_path, 'preprocess3D.csv')
    npy2csv_output_file = os.path.join(target_path, 'pred3D.csv')
    delt_dir = os.path.join(source_path, '3D')
    if not os.path.exists(pbb_dir):
        os.makedirs(pbb_dir)
        
###########################################################################################
        
    
    predict_3D(delt_dir, pbb_dir, model_weights_path) # write to pbb_output_path

    
###########################################################################################
    #pbb_dir = '/data/model_weights/crt/detect_result/tianchi_luna/20180507_model/'
    #delt_dir = '/data/model_weights/tlc/nodule_prep/tianchi_v3/'
    #path_uid_csv = '/home/crt/nodule_detector/annotations_tianchi_luna_pixel.csv'


    cads = pd.read_csv(path_uid_csv)[[u'path', u'uid']]
    path_list, uid_list = cads.values[:, 0], cads.values[:, 1]
    name_list = get_pbb_namelist(pbb_dir)

    new_data = []
    for name in name_list:
        #try:
            pbb = np.load(os.path.join(pbb_dir,"%s_pbb.npy"%name))
            delt_zyx = np.load(os.path.join(delt_dir, "%s_delta_zyx.npy"%name))
            ps, zs, ys, xs, ds = pbb[:,0], pbb[:,1], pbb[:,2], pbb[:,3], pbb[:,4]
            ps = ps[-ps.argsort()]
            ps = logit2prob(ps)
            zs = (zs + delt_zyx[0]) / delt_zyx[3] * delt_zyx[6]
            ys = (ys + delt_zyx[1]) / delt_zyx[4] * delt_zyx[7]
            xs = (xs + delt_zyx[2]) / delt_zyx[5] * delt_zyx[8]
            ds = ds * delt_zyx[7]
            candidates = np.column_stack((zs, ys, xs, ds, ps))
            candidates = candidates[candidates[:,4] > 0.3] # if comment, use all data
            clusters = bbox_dbscan(candidates, iou_eps=0.3, minPts=1)
            print 'cluster',clusters.shape
            cluster_coords = clusters[:, :5]
            cluster_ids = clusters[:, -1]
            candidates = []
            for i in np.arange(1, max(cluster_ids)+1):
                idxs = np.where(cluster_ids==i)[0]
                pts = np.array([cluster_coords[i] for i in idxs])
                #print 'pts',len(pts)
                centroid = get_centroid(pts)
                candidates.append(centroid)
            index = np.where(np.array(uid_list)==name)[0][0]
            image_path = path_list[index].encode('utf-8')
            name_dup = np.array([name] * len(candidates))
            image_path_dup = np.array([image_path] * len(candidates))
            candidates = np.column_stack((image_path_dup, name_dup, candidates))
            new_data.append(candidates)
        #except:
            #pass
    if new_data != []:
        new_data = np.vstack(new_data)
        df = pd.DataFrame(new_data, columns=[u'path', u'uid', u'coordZ', u'coordY', u'coordX', u'diameter', u'probability'])
    else:
        df = pd.DataFrame(columns=[u'path', u'uid', u'coordZ', u'coordY', u'coordX', u'diameter', u'probability'])
    df.to_csv(npy2csv_output_file, index=False)
#     name_list = get_pbb_namelist(pbb_dir) # np array of image uids 

#     cads = pd.read_csv(path_uid_csv)[[u'path', u'uid']]
#     path_list, uid_list = cads.values[:, 0], cads.values[:, 1]

#     all_candidates = []

#     for name in name_list:
#         try:
#             pbb = np.load(os.path.join(pbb_dir,"%s_pbb.npy"%name))
#             delt_zyx = np.load(os.path.join(delt_dir, "%s_delta_zyx.npy"%name))

#             ps, zs, ys, xs, ds = pbb[:,0], pbb[:,1], pbb[:,2], pbb[:,3], pbb[:,4]
#             ps = logit2prob(ps)
#             #print np.max(ps)
#             zs = (zs + delt_zyx[0]) / delt_zyx[3] * delt_zyx[6]
#             ys = (ys + delt_zyx[1]) / delt_zyx[4] * delt_zyx[7]
#             xs = (xs + delt_zyx[2]) / delt_zyx[5] * delt_zyx[8]
#             ds = ds * delt_zyx[7]
#             candidates = np.column_stack((ps, zs, ys, xs, ds))
#             candidates = candidates[candidates[:,0] > prob_thre]
#             #print candidates

#             kept_idxs = nms_inner(candidates)

#             candidates = np.array([candidates[i] for i in kept_idxs])

#             index = np.where(np.array(uid_list)==name)[0][0]
#             image_path = path_list[index].encode('utf-8')

#             name_dup = np.array([name] * len(candidates))
#             image_path_dup = np.array([image_path] * len(candidates))

#             candidates = np.column_stack((name_dup, image_path_dup, candidates))
            
#             if len(candidates) > 0:
#                 all_candidates.append(candidates)
#         except Exception, err:
#             print err


#     #print all_candidates
#     if len(all_candidates):
#         all_candidates = np.vstack(all_candidates)
#     #print all_candidates
#         candidate_csv = pd.DataFrame(all_candidates, columns=['uid', 'path', 'probability', 'coordZ', 'coordY',\
#                                             'coordX', 'diameter'])
#     else:
#         candidate_csv = pd.DataFrame(columns=['uid', 'path', 'probability', 'coordZ', 'coordY',\
#                                             'coordX', 'diameter'])
#     candidate_csv.to_csv(npy2csv_output_file, index=False)