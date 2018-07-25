import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import SimpleITK as sitk
import cv2
from tqdm import tqdm 
from utils.manifest import *

def bboxes_iou(bbox1, bbox2):
    '''iou of two 3d bboxes. bbox: [z,y,x,d]'''
    
    # zmin, zmax, ymin, ymax, xmin, xmax
    bbox1 = [bbox1[0]-bbox1[3]/2, bbox1[0]+bbox1[3]/2,  
             bbox1[1]-bbox1[3]/2, bbox1[1]+bbox1[3]/2, 
             bbox1[2]-bbox1[3]/2, bbox1[2]+bbox1[3]/2]
    bbox2 = [bbox2[0]-bbox2[3]/2, bbox2[0]+bbox2[3]/2, 
             bbox2[1]-bbox2[3]/2, bbox2[1]+bbox2[3]/2, 
             bbox2[2]-bbox2[3]/2, bbox2[2]+bbox2[3]/2]
    
    # Intersection bbox and volume.
    int_zmin = np.maximum(bbox1[0], bbox2[0])
    int_zmax = np.minimum(bbox1[1], bbox2[1])
    int_ymin = np.maximum(bbox1[2], bbox2[2])
    int_ymax = np.minimum(bbox1[3], bbox2[3])
    int_xmin = np.maximum(bbox1[4], bbox2[4])
    int_xmax = np.minimum(bbox1[5], bbox2[5])

    int_z = np.maximum(int_zmax - int_zmin, 0.)
    int_y = np.maximum(int_ymax - int_ymin, 0.)
    int_x = np.maximum(int_xmax - int_xmin, 0.)
    
    int_vol = int_z * int_y * int_x
    
    vol1 = (bbox1[1] - bbox1[0]) * (bbox1[3] - bbox1[2]) * (bbox1[5] - bbox1[4])
    vol2 = (bbox2[1] - bbox2[0]) * (bbox2[3] - bbox2[2]) * (bbox2[5] - bbox2[4])
    iou = int_vol / (vol1 + vol2 - int_vol)
    return iou

def distance(vx1, vx2):
    return np.linalg.norm(np.array(vx1)-np.array(vx2))
    

def bbox_dbscan(candidates, iou_eps=0.3, minPts=1, use_prob='det'):
    '''candidates: a numpy array, columns are z, y, x, d, (det_p, cls_p). candidates of a SINGLE ct!
    iou_eps: if larger than iou_eps -> included in one cluster, which is opposite to the common eps
    '''
    ori_candidates = candidates.copy()
    candidates = candidates[:, :4]
    checked = np.zeros(len(candidates), dtype=np.bool_)
    cluster_id = np.zeros(len(candidates), dtype=np.int)
    core_pt = np.zeros(len(candidates), dtype=np.bool_)
    curr_id = 0
    
    while np.sum(checked) < len(checked):
        if np.sum(checked) < np.sum(cluster_id.astype(np.bool_)):
            cand_idxs = np.where(np.logical_xor(checked, cluster_id.astype(np.bool_)))[0]
            for cand_idx in cand_idxs: 
                if core_pt[cand_idx]:
                    temp_i = []
                    for i in range(len(candidates)):
                        if checked[i]==0 and cluster_id[i]==0 and bboxes_iou(candidates[cand_idx], candidates[i]) > iou_eps: 
                            cluster_id[i] = curr_id
                            temp_i.append(i)
                    if len(temp_i) >= minPts-1:
                        for j in temp_i:
                            core_pt[j] = 1
                checked[cand_idx] = 1
        else:  
            curr_id += 1
            start_idx = np.where(cluster_id==0)[0][0]
            cluster_id[start_idx] = curr_id
            core_pt[start_idx] = 1
    return np.column_stack([ori_candidates, cluster_id])


def get_centroid(candidates, use_prob='det', diam='max', prob='max'):
    '''candidates:  [z, y, x, d, det_p, (cls_p)], diam = 'max' or 'mean', prob = 'mean' or 'max',
    use_prob = 'det', 'cls', 'both', or None
    TODO!!!: add weights and d expansion 
    '''
    # calculate the weights if use probability as the weights
    if use_prob: # weighted mean
        if use_prob == 'det':
            probs = candidates[:, 4]
        elif use_prob == 'cls':
            try:
                probs = candidates[:, 5]
            except:
                probs = candidates[:, 4]
                print 'classification probability not available, using detection probability ... '
        else: # both
            try:
                probs = (candidates[:, 4] + candidates[:, 5]) / 2.
            except:
                probs = candidates[:, 4]
                print 'classification probability not available, using detection probability ... '
        probs = probs * 1. / sum(probs)
    
    # calculate the centroid
    if use_prob:
        centroid = np.dot(probs, candidates[:, :3])
    else: # mean
        centroid = np.mean(candidates[:, :3], axis=0)
    
    # calculate the diameter
    if diam == 'max': # find max diameter
        mins = [candidate[:3] - candidate[3] for candidate in candidates]
        maxs = [candidate[:3] + candidate[3] for candidate in candidates]
        d = np.max(np.array(maxs) - np.array(mins))
    else:
        if use_prob: # weighted mean
            d = np.dot(probs, candidates[:, 3])
        else:
            d = np.mean(candidates[:, 3])
    
    # calculate probability
    if prob == 'max':
        det_p = np.max(candidates[:, 4])
        if candidates.shape[1] >= 6: # has cls prob
            cls_p = np.max(candidates[:, 5])
    else:
        det_p = np.mean(candidates[:, 4])
        if candidates.shape[1] > 5:  # has cls prob
            cls_p = np.mean(candidates[:, 5])
    
    # concatenate the result
    try:
        out = np.concatenate((centroid, [d], [det_p], [cls_p]))
    except:
        out = np.concatenate((centroid, [d], [det_p]))
        
    assert candidates.shape[1] == len(out)
    
    return out


def dbscan(input_csv_path, output_csv_path, iou_thresh=0.1, minpts=3,  
    column_names=[u'uid', u'path', u'coordZ', u'coordY', u'coordX', u'diameter',\
                         u'det_probability', u'cls_probability']):
    predictions_filename = input_csv_path
    pred = pd.read_csv(predictions_filename)[column_names]
    #print pred.values.shape
    sids = list(set(pred.values[:,0]))
    
    new_data = []
    total = len(sids)
    for i in range(total):
        predictions = pred[pred[u'uid'] == sids[i]].values
        sid = predictions[0][0]
        path = predictions[0][1]
        # z y x d
        predictions = predictions[:, 2:]
        clusters = bbox_dbscan(predictions, iou_eps=iou_thresh, minPts=minpts)
        cluster_coords = clusters[:,:6]
        cluster_ids = clusters[:, -1]
        candidates = []
        for j in np.arange(1, max(cluster_ids)+1):
            idxs = np.where(cluster_ids==j)[0]
            pts = np.array([cluster_coords[k] for k in idxs])
            centroid = get_centroid(pts)
            candidates.append(centroid)
        path_dup = np.array([path] * len(candidates))
        sid_dup = np.array([sid] * len(candidates))
        candidates = np.column_stack([sid_dup, path_dup, candidates])
        new_data.extend(candidates)
        print_percentage('dbscan', i + 1, total)
    
    if new_data != []:
        new_data = np.array(new_data)
        #print new_data.shape
        df = pd.DataFrame(new_data, columns = [u'uid', u'path', u'coordZ', u'coordY', u'coordX', u'diameter',\
                             u'det_probability', u'cls_probability'])
    else:
        df = pd.DataFrame(columns = [u'uid', u'path', u'coordZ', u'coordY', u'coordX', u'diameter',\
                             u'det_probability', u'cls_probability'])
    
    df.to_csv(output_csv_path, index = False)
    
        

    #return df

