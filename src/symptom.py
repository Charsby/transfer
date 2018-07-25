import os
import sys
import numpy as np
import scipy.ndimage as nd
import keras
import keras.backend as K
from keras.models import load_model, Model
from keras.applications.vgg16 import VGG16
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.core import Dense

def get_symptom_classifier(model_path):
    base_model = VGG16(weights=None, include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(3, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    model.load_weights(model_path)
    return model
    

def symptom_classification(model, image, centroid, diameter, spacing):
    cor_x = centroid[2]
    cor_y = centroid[1]
    cor_z = centroid[0]
    #print image.shape
    #print centroid
    #print spacing
    #print diameter
    if float(diameter)< 10.0:
        margin = int((float(diameter)/min(spacing[1], spacing[2])))+10
    else:
        margin = int((float(diameter)/min(spacing[1], spacing[2]))/2)+10
    
    #print image.shape
    image_crop = image[cor_z, cor_y-margin:cor_y+margin, cor_x-margin:cor_x+margin]
    #print image_crop.shape
    resize_shape = np.divide((48.0, 48.0), image_crop.shape)
    # print resize_shape
    # print np.any(np.isnan(image_crop))
    image_resize = nd.interpolation.zoom(image_crop, resize_shape, mode = 'nearest')
    
    candidate = image_resize
    image_candi = np.expand_dims(candidate, axis = 2)
    
    temp = np.dstack((image_candi, image_candi))
    image_3c = np.dstack((temp, image_candi))
    mean = -618.9659089302514
    std = 405.28599607414463
    test_final = (image_3c - mean) / std
    test_final = test_final[np.newaxis, ...]
    test_pred = model.predict(test_final)
    #symptom = np.argmax(test_pred)
    return test_pred