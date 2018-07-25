#**********************************

from utils.gpu_allocation import set_gpu
num_gpu = 1
set_gpu(num_gpu,gpu_list=[0,1,2,3])
import keras.backend as K
K.set_learning_phase(0)
from src.dicom_reader import read_in
from src.process_2D import process_2d
from src.segmentation3D import postprocess
from src.process_3D import process_3d
import sys
import os
import shutil

def full_process(path_to_ct, tmp_folder='/data2/model_weights/zwq/test/'):
    
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder)
    
    preprocess_dir = os.path.join(tmp_folder, 'preprocess')
    pred_dir = os.path.join(tmp_folder, 'pred')
    
    #readin dicom files and preprocess(path -> path)
    read_in(path_to_ct, preprocess_dir)
    #saved npy  ./test/preprocess/temp2D ./test/preprocess/temp3D
    
    #processing thick CT with 2d detector and classifier
    process_2d(preprocess_dir, pred_dir,\
              ssd_model_path = '/data2/model_weights/zwq/ssd/final/v3/ssd_300_vgg-200',\
              classifier_model_path = '/data2/model_weights/fyx/nodule_classifier_48p719_05-0.95.hdf5')
    
    #processing thin CT with 3d detector and classifier
    process_3d(preprocess_dir, pred_dir,\
              detector_model_path = '/data2/model_weights/crt/all_epoch30_valloss0.32.hdf5',\
              classifier_model_path = '/data2/test/nodule_classification/weights_wrnet_ccyy.30-0.9351-0.2998.hdf5')
    
    #processing 3d segmentation and calculating diameter
    result = postprocess(pred_dir,\
            symptom_classifier_path = '/data2/model_weights/fyx/symptom_classifier_vgg_48p712_17-0.89.hdf5')
    
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    
    return result