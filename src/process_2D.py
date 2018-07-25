from ssd_detector import ssd_detection
from two_d_seg import two_d_seg_infer
from two_d_classify import classifier_2d_nodule
import os
from utils.manifest import *
from utils.dbscan import dbscan
import shutil

def process_2d(source_path, target_path, ssd_model_path, classifier_model_path, select_thre = 0.2, nms_thre = 1):
    
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    os.makedirs(target_path)

    
    print_title('start 2d detection!')
    
    ssd_detection(source_path, target_path, ssd_model_path, select_thre, nms_thre)
 
    print_gap()
    
    dbscan(os.path.join(target_path, 'pred2D.csv'), os.path.join(target_path, 'pred2D_c.csv'))

    print_title('start 2d classification!')
    
    classifier_2d_nodule(MODEL_PATH = classifier_model_path, TARGET_PATH = target_path, OUTPUT_DIM = [48,48])
    
    print_gap()