import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from keras.models import load_model
import tensorflow as tf
import keras
from utils.preprocessing.preprocess_utils import load_CT
from utils.manifest import *
import keras.backend as K
from keras.models import load_model, Model
from keras.applications.vgg16 import VGG16
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.core import Dense

def get_classifier(model_path):
    base_model = VGG16(weights=None, include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    model.load_weights(model_path)
    return model

def classifier_2d_nodule(MODEL_PATH, TARGET_PATH, OUTPUT_DIM = [48, 48]):
    # read from csv
    # tf.reset_default_graph()
    TEST_PATH = os.path.join(TARGET_PATH, 'pred2D_c.csv')
    pred_df = pd.read_csv(TEST_PATH)
    pred_col = pred_df.columns.values.tolist()
    pred_img_path = pred_df['path']
    pred_x = pred_df['coordX']
    pred_y = pred_df['coordY']
    pred_z = pred_df['coordZ']
    pred_diameter = pred_df['diameter']
    pred_prob = pred_df['det_probability']

    path_list = list(pred_df['path'].drop_duplicates())
    test_data = []
    pd_data_list = []
    idx = -1
    
    total = len(path_list)
    for path in path_list:
        idx += 1
        #image, _ = ReadDICOMFolder(path)
        image, _, _, uid = load_CT(path)
        #image = np.flip(image, 0)
        x_len = image.shape[2] # column
        y_len = image.shape[1] # row

        bbox_pred_idx = pred_df[pred_df['path'] == path].index.tolist()
        num_pred = len(bbox_pred_idx)
        for i in range(num_pred):
            # crop part
            info_pred = [pred_x.loc[bbox_pred_idx[i]], pred_y.loc[bbox_pred_idx[i]], pred_z.loc[bbox_pred_idx[i]]]
            sample_prob = pred_prob.loc[bbox_pred_idx[i]]
            sample_uid = uid
            sample_d = pred_diameter.loc[bbox_pred_idx[i]]
            cor_x = int(float(info_pred[0]))
            cor_y = int(float(info_pred[1]))
            cor_z = int(float(info_pred[2]))

            # cropping
            candidate = image[cor_z]
            offset = [OUTPUT_DIM[0]/2, OUTPUT_DIM[1]/2]
            result = np.zeros(OUTPUT_DIM, dtype=np.int16) - 1000
            #try:
            for row in range(OUTPUT_DIM[0]):
                for col in range(OUTPUT_DIM[1]):
                    if (cor_x-offset[0]+col >= x_len) or (cor_y-offset[1]+row >= y_len):
                        result[row, col] = 0
                    else:
                        result[row, col] = candidate[cor_y - offset[0] + row, cor_x - offset[1] + col]
            test_data.append(result)
            pd_data = [sample_uid, path, cor_x, cor_y, cor_z, sample_d, sample_prob]
            pd_data_list.append(pd_data)
        #print_percentage('Classification', idx + 1, total)

    test_data = np.array(test_data)
    test_data = np.expand_dims(test_data, axis = 3)

    test_data = (test_data - 565.6166237728834) / 420.60462967616536

    test_final = np.zeros(shape = [test_data.shape[0], 48, 48, 3])
    for i in range(test_data.shape[0]):
        test_final[i, :, :, 0] = test_data[i, :, :, 0]
        test_final[i, :, :, 1] = test_data[i, :, :, 0]
        test_final[i, :, :, 2] = test_data[i, :, :, 0]
    
    K.clear_session()
    model = get_classifier(MODEL_PATH)
    # test the model
    
###########################################################################################


    test_pred = model.predict(test_final)

    
###########################################################################################    
    
    if len(test_pred):
        test_prob = test_pred[:,1]
        #if len(test_pred.shape) == 1:
        #    test_pred = test_pred[np.newaxis, ...]
        res = np.argmax(test_pred, axis = 1)
        num = len(res)
        pd_result = []
        cls_prob = []
        #print len(pd_data_list[0])
        #print test_pred.shape
        #print test_prob[0]
        for i in range(num):
            if res[i] == 1:
                pd_result.append(pd_data_list[i])
                cls_prob.append([test_prob[i]])
            else:
                continue
        final_result = np.hstack((pd_result,cls_prob))
        #print final_result.shape
        df_result = pd.DataFrame(final_result,columns=['uid', 'path', 'coordX', 'coordY', 'coordZ', 'diameter', 'det_probability', 'cls_probability'])
    else:
        df_result = pd.DataFrame(columns=['uid', 'path', 'coordX', 'coordY', 'coordZ', 'diameter', 'det_probability', 'cls_probability'])
    df_result.to_csv(os.path.join(TARGET_PATH, 'pred2D_classified.csv'),index=False)
    #return df_result