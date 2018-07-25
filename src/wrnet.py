from keras.models import Sequential, Model
from keras.layers import Input, add
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D, ZeroPadding3D
from keras.layers.pooling import AveragePooling3D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD
from keras import backend as K
from keras.utils import plot_model
import keras
#import pydot
#import graphviz
import numpy as np
import h5py
import os


def get_model(summary=False):
    #model = Sequential()
    
    main_input = Input(shape=(48,48,48,1), name='main_input')
    bn0 = BatchNormalization(name='bn0')(main_input)
    conv1_1 = Conv3D(filters = 16, kernel_size = (3, 3, 3), strides = (1, 1, 1), 
                     padding = 'same', name='conv1_1', kernel_initializer='he_normal')(bn0)     
    bn1_1 = BatchNormalization(scale=True,name='bn1_1')(conv1_1) 
    relu1_1 = LeakyReLU(alpha=0.01, name='relu1_1')(bn1_1)
    
    # 1st layer group
    
    conv1_2 = Conv3D(filters = 32, kernel_size = (3, 3, 3), strides = (1, 1, 1), 
                     padding = 'same', name='conv1_2', kernel_initializer='he_normal')(relu1_1)     
    bn1_2 = BatchNormalization(scale=True,name='bn1_2')(conv1_2) 
    relu1_2 = LeakyReLU(alpha=0.01, name='relu1_2')(bn1_2)
    dropout1_1 = Dropout(.5, name='dropout1_1')(relu1_2)
    conv1_3 = Conv3D(filters = 32, kernel_size = (3, 3, 3), strides = (1, 1, 1), 
                     padding = 'same', name='conv1_3', kernel_initializer='he_normal')(dropout1_1)    
    conv1_4 = Conv3D(filters = 32, kernel_size = (3, 3, 3), strides = (1, 1, 1), 
                     padding = 'same', name='conv1_4', kernel_initializer='he_normal')(relu1_1)    
    eltwise1 = add([conv1_3, conv1_4])
    
    bn1_3 = BatchNormalization(scale=True,name='bn1_3')(eltwise1) 
    relu1_3 = LeakyReLU(alpha=0.01, name='relu1_3')(bn1_3)
    conv1_5 = Conv3D(filters = 32, kernel_size = (3, 3, 3), strides = (1, 1, 1), 
                     padding = 'same', name='conv1_5', kernel_initializer='he_normal')(relu1_3) 
    bn1_4 = BatchNormalization(scale=True,name='bn1_4')(conv1_5) 
    relu1_4 = LeakyReLU(alpha=0.01, name='relu1_4')(bn1_4)
    conv1_6 = Conv3D(filters = 32, kernel_size = (3, 3, 3), strides = (1, 1, 1), 
                     padding = 'same', name='conv1_6', kernel_initializer='he_normal')(relu1_4) 
    eltwise2 = add([conv1_6, eltwise1])
    
    #bn3 = BatchNormalization(scale=True,name='bn3')(eltwise1) 
    #relu3 = LeakyReLU(alpha=0.01, name='relu3')(bn3)
    
    
    # 2nd layer group
    bn2_1 = BatchNormalization(scale=True,name='bn2_1')(eltwise2) 
    relu2_1 = LeakyReLU(alpha=0.01, name='relu2_1')(bn2_1)
    
    conv2_2 = Conv3D(filters = 64, kernel_size = (3, 3, 3), strides = (2, 2, 2), 
                     padding = 'same', name='conv2_2', kernel_initializer='he_normal')(relu2_1)     
    bn2_2 = BatchNormalization(scale=True,name='bn2_2')(conv2_2) 
    relu2_2 = LeakyReLU(alpha=0.01, name='relu2_2')(bn2_2)
    dropout2_1 = Dropout(.2, name='dropout2_1')(relu2_2)
    conv2_3 = Conv3D(filters = 64, kernel_size = (3, 3, 3), strides = (1, 1, 1), 
                     padding = 'same', name='conv2_3', kernel_initializer='he_normal')(dropout2_1)    
    conv2_4 = Conv3D(filters = 64, kernel_size = (3, 3, 3), strides = (2, 2, 2), 
                     padding = 'same', name='conv2_4', kernel_initializer='he_normal')(relu2_1)    
    eltwise3 = add([conv2_3, conv2_4])
    
    bn2_3 = BatchNormalization(scale=True,name='bn2_3')(eltwise3) 
    relu2_3 = LeakyReLU(alpha=0.01, name='relu2_3')(bn2_3)
    conv2_5 = Conv3D(filters = 64, kernel_size = (3, 3, 3), strides = (1, 1, 1), 
                     padding = 'same', name='conv2_5', kernel_initializer='he_normal')(relu2_3) 
    bn2_4 = BatchNormalization(scale=True,name='bn2_4')(conv2_5) 
    relu2_4 = LeakyReLU(alpha=0.01, name='relu2_4')(bn2_4)
    conv2_6 = Conv3D(filters = 64, kernel_size = (3, 3, 3), strides = (1, 1, 1), 
                     padding = 'same', name='conv2_6', kernel_initializer='he_normal')(relu2_4) 
    eltwise4 = add([conv2_6, eltwise3])
    
    # 3rd layer group
    bn3_1 = BatchNormalization(scale=True,name='bn3_1')(eltwise4) 
    relu3_1 = LeakyReLU(alpha=0.01, name='relu3_1')(bn3_1)
    
    conv3_2 = Conv3D(filters = 128, kernel_size = (3, 3, 3), strides = (2, 2, 2), 
                     padding = 'same', name='conv3_2', kernel_initializer='he_normal')(relu3_1)     
    bn3_2 = BatchNormalization(scale=True,name='bn3_2')(conv3_2) 
    relu3_2 = LeakyReLU(alpha=0.01, name='relu3_2')(bn3_2)
    dropout3_1 = Dropout(.2, name='dropout3_1')(relu3_2)
    conv3_3 = Conv3D(filters = 128, kernel_size = (3, 3, 3), strides = (1, 1, 1), 
                     padding = 'same', name='conv3_3', kernel_initializer='he_normal')(dropout3_1)    
    conv3_4 = Conv3D(filters = 128, kernel_size = (3, 3, 3), strides = (2, 2, 2), 
                     padding = 'same', name='conv3_4', kernel_initializer='he_normal')(relu3_1)    
    eltwise5 = add([conv3_3, conv3_4])
    
    bn3_3 = BatchNormalization(scale=True,name='bn3_3')(eltwise5) 
    relu3_3 = LeakyReLU(alpha=0.01, name='relu3_3')(bn3_3)
    conv3_5 = Conv3D(filters = 128, kernel_size = (3, 3, 3), strides = (1, 1, 1), 
                     padding = 'same', name='conv3_5', kernel_initializer='he_normal')(relu3_3) 
    bn3_4 = BatchNormalization(scale=True,name='bn3_4')(conv3_5) 
    relu3_4 = LeakyReLU(alpha=0.01, name='relu3_4')(bn3_4)
    conv3_6 = Conv3D(filters = 128, kernel_size = (3, 3, 3), strides = (1, 1, 1), 
                     padding = 'same', name='conv3_6', kernel_initializer='he_normal')(relu3_4) 
    eltwise6 = add([conv3_6, eltwise5])
    
    
    # 4th layer group
    bn4_1 = BatchNormalization(scale=True,name='bn4_1')(eltwise6) 
    relu4_1 = LeakyReLU(alpha=0.01, name='relu4_1')(bn4_1)
    
    conv4_2 = Conv3D(filters = 256, kernel_size = (3, 3, 3), strides = (2, 2, 2), 
                     padding = 'same', name='conv4_2', kernel_initializer='he_normal')(relu4_1)     
    bn4_2 = BatchNormalization(scale=True,name='bn4_2')(conv4_2) 
    relu4_2 = LeakyReLU(alpha=0.01, name='relu4_2')(bn4_2)
    dropout4_1 = Dropout(.2, name='dropout4_1')(relu4_2)
    conv4_3 = Conv3D(filters = 256, kernel_size = (3, 3, 3), strides = (1, 1, 1), 
                     padding = 'same', name='conv4_3', kernel_initializer='he_normal')(dropout4_1)    
    conv4_4 = Conv3D(filters = 256, kernel_size = (3, 3, 3), strides = (2, 2, 2), 
                     padding = 'same', name='conv4_4', kernel_initializer='he_normal')(relu4_1)    
    eltwise7 = add([conv4_3, conv4_4])
    
    bn4_3 = BatchNormalization(scale=True,name='bn4_3')(eltwise7) 
    relu4_3 = LeakyReLU(alpha=0.01, name='relu4_3')(bn4_3)
    conv4_5 = Conv3D(filters = 256, kernel_size = (3, 3, 3), strides = (1, 1, 1), 
                     padding = 'same', name='conv4_5', kernel_initializer='he_normal')(relu4_3) 
    bn4_4 = BatchNormalization(scale=True,name='bn4_4')(conv4_5) 
    relu4_4 = LeakyReLU(alpha=0.01, name='relu4_4')(bn4_4)
    conv4_6 = Conv3D(filters = 256, kernel_size = (3, 3, 3), strides = (1, 1, 1), 
                     padding = 'same', name='conv4_6', kernel_initializer='he_normal')(relu4_4) 
    eltwise8 = add([conv4_6, eltwise7])
    # remaining parts
    bn5_1 = BatchNormalization(scale=True,name='bn5_1')(eltwise8) 
    relu5_1 = LeakyReLU(alpha=0.01, name='relu5_1')(bn5_1)
    pool1 = MaxPooling3D(pool_size=(3, 5, 5), strides=None, padding='valid', data_format=None)(relu5_1)
    flatten1 = Flatten()(pool1)
    #fc1 = Dense(64,activation='softmax')(flatten1)
    fc2 = Dense(2,activation='softmax')(flatten1)
    
    model = Model(input=main_input, output=fc2)
    if summary:
        print(model.summary())
    return model