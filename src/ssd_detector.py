import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2
#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
#sys.path.append('/home/wrj/deeplearning/')
#sys.path.append('/home/zwq/nodule/preprocessing')
import tensorflow as tf
import utils.preprocessing
from utils.object_detection import nets_factory
from utils.object_detection import nets_factory,np_methods
from utils.preprocessing import ssd_vgg_preprocessing
import utils.visualization
import pandas as pd
from utils.manifest import *
#import evaluation
slim = tf.contrib.slim

def get_data_path(cads, uid):
    return cads[cads[u'uid'] == uid.decode('utf-8')][u'path'].values[0].encode('utf-8')
def process_image(img, isess, img_input, image_4d, predictions, \
                  localisations, bbox_img, ssd_anchors,\
                  select_threshold=0.2, nms_threshold=0.1, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=2, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes

def ssd_detection(s_path, target_path, model_path, select_threshold = 0.2, nms_threshold = 0.1):
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    isess = tf.InteractiveSession(config=config)
    
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    
    model_class = nets_factory.get_network('ssd_300_vgg') #get the net class 
    model_params = model_class.default_params._replace(num_classes = 2, no_annotation_label = 2)
    model_net = model_class(model_params) # get the net structure 
    model_shape = model_net.params.img_shape # img shape for the net 
    model_anchors = model_net.anchors(model_shape)
    
    # Input placeholder.
    net_shape = (300, 300)
    data_format = 'NHWC'
    img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
    # Evaluation pre-processing: resize to SSD net shape.
    image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
        img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
    image_4d = tf.expand_dims(image_pre, 0)

    # Define the SSD model.
    reuse = True if 'ssd_net' in locals() else None
    with slim.arg_scope(model_net.arg_scope(data_format=data_format)):
        predictions, localisations, _, _ = model_net.net(image_4d, is_training=False, reuse=reuse)
        
    # Restore SSD model.
    ckpt_filename = model_path
    # ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
    isess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(isess, ckpt_filename)

    # SSD default anchor boxes.
    ssd_anchors = model_net.anchors(net_shape)
    
    #start processing!
    pred_path = []
    pred_uid = []
    pred_z = []
    pred_y = []
    pred_x = []
    pred_d = []
    pred_p = []
    pred_cp = []
    cads = pd.read_csv(os.path.join(s_path, 'preprocess2D.csv'))
    source_path = os.path.join(s_path, '2D')
    total = len(os.listdir(source_path))
    for index, file_name in enumerate(os.listdir(source_path)):
        if '_clean' in file_name:
            #read in images to process
            uid = file_name.split('_')[0]
            s_path = get_data_path(cads,uid)
            data = np.load(os.path.join(source_path, file_name))
            delta = np.load(os.path.join(source_path, file_name).replace('clean','delta_zyx'))
            
            for i in range(data.shape[1]):
                img = data[0, i, ...]
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                #process each layer
                ###########################################################################################
                
                
                rclasses, rscores, rbboxes = process_image(img, isess, img_input, image_4d, predictions, localisations, bbox_img, ssd_anchors)
                
                
                ###########################################################################################
                for j, bbox in enumerate(rbboxes):
                    #convert bbox to centroid and diameter
                    p_z = i
                    p_y = (bbox[0] + bbox[2]) * data.shape[2] / 2.0
                    p_x = (bbox[1] + bbox[3]) * data.shape[3] / 2.0
                    p_d = np.maximum((bbox[2] - bbox[0]) * data.shape[2], (bbox[3] - bbox[1]) * data.shape[3])
                    #convert preprocessed coordinates to original coordinates
                    pred_cord = (np.array([p_z, p_y, p_x]) + delta[:3]) * delta[6:9] / delta[3:6]
                    pred_cord_z = pred_cord[0]
                    pred_cord_y = pred_cord[1]
                    pred_cord_x = pred_cord[2]
                    pred_dia = p_d * delta[7]
                    #collect and save
                    pred_z.append(pred_cord_z)
                    pred_y.append(pred_cord_y)
                    pred_x.append(pred_cord_x)
                    pred_d.append(pred_dia)
                    pred_uid.append(uid)
                    pred_p.append(rscores[j])
                    pred_path.append(s_path)
                    pred_cp.append(0)
            print_percentage('process2D', index + 1, total, uid + 'finished!')
    
    #save as csv file
    pred_uid = np.array(pred_uid)
    pred_path = np.array(pred_path)
    pred_z = np.array(pred_z)
    pred_y = np.array(pred_y)
    pred_x = np.array(pred_x)
    pred_d = np.array(pred_d)
    pred_p = np.array(pred_p)
    pred_cp = np.array(pred_cp)

    df = pd.DataFrame(np.column_stack([pred_uid, pred_path, pred_z, pred_y, pred_x, pred_d, pred_p, pred_cp]),
                     columns = ['uid', 'path', 'coordZ', 'coordY', 'coordX', 'diameter', 'det_probability', 'cls_probability'])
    df.to_csv(os.path.join(target_path,'pred2D.csv'), index = False)
    
    isess.close()
    tf.reset_default_graph()