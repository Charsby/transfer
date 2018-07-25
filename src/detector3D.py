import os
import time
import numpy as np
from importlib import import_module
from glob import glob
from functools import partial
from math import cos, pi
from grt123 import GRT123Net
from loss import Loss
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.layers.core import Lambda
from keras.utils import plot_model
import torch
from torch.utils.data import DataLoader
from keras.layers import merge
import data_gen
from utils.split_combine import SplitComb
from keras.models import Model
import keras.backend as K
from utils.manifest import *

config = {}
config['anchors'] = [ 3.0, 10.0, 30.0]
config['chanel'] = 1 
config['crop_size'] = [128, 128, 128]
config['stride'] = 4 #crop patch, coord input shape=(32,32,32)
config['max_stride'] = 16 #crop patch
config['num_neg'] = 800 #number of neg samples when training 
config['th_neg'] = 0.02 #selectsample bbox and anchor iou thresh 
config['th_pos_train'] = 0.5 #
config['th_pos_val'] = 1 #
config['num_hard'] = 2 
config['bound_size'] = 12 #crop
config['reso'] = 1 #spacing
config['sizelim'] = 3. # mm, hard example mining
config['sizelim2'] = 50 # 
config['sizelim3'] = 100 # 
config['aug_scale'] = True # scale augmentation
config['r_rand_crop'] = 0.3 # train [1/(1-0.3)-1]*100% sample
config['pad_value'] = 170 # cropping
config['max_diameter'] = 50. # 
config['augtype'] = {'flip':True,'swap':False,'scale':True,'rotate':False} # augs
config['blacklist'] = []

def get_namelist(dir, order='train_first'):
    samples = glob(os.path.join(dir, "*_clean.npy"))
    samples = sorted(samples)
    return np.array(samples)

def get_pbb(output, thresh, ismask):
    stride = 4
    anchors = np.asarray([ 3.0, 10.0, 30.0])

    output = np.copy(output)
    offset = (float(stride) - 1) / 2
    output_size = output.shape
    # output[0] is logit, output[1:3] are delta(xyz)/xyz, output[4] is log(delta(r))
    oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
    oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
    ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)

    output[:, :, :, :, 1] = oz.reshape((-1, 1, 1, 1)) + output[:, :, :, :, 1] * anchors.reshape((1, 1, 1, -1))
    output[:, :, :, :, 2] = oh.reshape((1, -1, 1, 1)) + output[:, :, :, :, 2] * anchors.reshape((1, 1, 1, -1))
    output[:, :, :, :, 3] = ow.reshape((1, 1, -1, 1)) + output[:, :, :, :, 3] * anchors.reshape((1, 1, 1, -1))
    output[:, :, :, :, 4] = np.exp(output[:, :, :, :, 4]) * anchors.reshape((1, 1, 1, -1))
    mask = output[..., 0] > thresh
    xx,yy,zz,aa = np.where(mask)

    output = output[xx,yy,zz,aa]
    if ismask:
        return output,[xx,yy,zz,aa]
    else:
        return output
    
def predict_3D(datadir, outdir, weight_path):
    namelist = get_namelist(datadir)

    batch_size = 1
    workers = 4
    # get the net
    K.clear_session()
    net = GRT123Net()
    model = net.get_model(input_tensor = None, input_shape = (1, 128, 128, 128),
        input_channel = 1, output_channel = 3, coord = None)
    kloss = Loss()
    adam = keras.optimizers.SGD(lr=0.00001)
    model.compile(loss = kloss.ohnm_loss, optimizer=adam)
    model.load_weights(weight_path)

    margin = 16
    sidelen = 96

    split_comber = SplitComb(sidelen, config['max_stride'], config['stride'],
                             margin, config['pad_value'])

    dataset = data_gen.DataBowl3Detector_ori(
        #val_namelist,
        namelist,
        config,
        phase='test',
        split_comber=split_comber)

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=data_gen.collate,
        pin_memory=False)

    split_comber = test_loader.dataset.split_comber
    total = len(test_loader)
    for i_name, (data, target, coord, nzhw) in enumerate(test_loader):
        s = time.time()
        target = [np.asarray(t, np.float32) for t in target]
        nzhw = nzhw[0]
        name = test_loader.dataset.filenames[i_name].split('/')[-1]
        shortname = name.split('_clean')[0]
        data = data[0][0]
        coord = coord[0][0] 
        isfeat = False
        if 'output_feature' in config:
            if config['output_feature']:
                isfeat = True
        n_per_run = 2
        splitlist = range(0,len(data)+1,2)
        if splitlist[-1]!=len(data):
            splitlist.append(len(data))
        outputlist = []
        featurelist = []

        for i in range(len(splitlist)-1):
            input = data[splitlist[i]:splitlist[i+1]].numpy()
            inputcoord = coord[splitlist[i]:splitlist[i+1]].numpy()
            
###########################################################################################


            output = model.predict([input,inputcoord])
            
        
###########################################################################################            
            outputlist.append(output)
        #print '-'*50
        output = np.concatenate(outputlist,0)
        output = split_comber.combine(output,nzhw=nzhw)

        thresh = -2.2  # logit>-2.2 => probability>0.1; logit>-1.4 => probability>0.2
        pbb,mask = get_pbb(output,thresh,ismask=True)
        if isfeat:
            feature_selected = feature[mask[0],mask[1],mask[2]]
            np.save(os.path.join(outdir, shortname+'_feature.npy'), feature_selected)

        #print([i_name,shortname])

        np.save(os.path.join(outdir, shortname+'_pbb.npy'), pbb)
        print_percentage('Detection3D', i_name + 1, total)
    print_gap()