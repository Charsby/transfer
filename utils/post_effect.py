import matplotlib.pyplot as plt
import keras
from keras.models import load_model 
import numpy as np
import pandas as pd
import os
from scipy import misc
from keras import backend as K
import skimage.transform as transform

    
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def heatmap(model,
            image,
            resized_l=224):
    """
    A function that generate heatmap for a given image using the pasing model
    
    Args:
        model: the model to use
        image: the image that need to be tested 
    
    Returns:
        a resized heatmap
    
    """
    # get layer dictionary first
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    # get the activition map
    inp = model.input 
    x = layer_dict['activation_49'].output
    get_output = K.function([inp], [x]) 
    activition_map = np.array(get_output([image]))[0][0]
#     print activition_map.shape

    # get the weight of the fc layer
    weights_all = np.array(layer_dict['dense_1'].get_weights()[0])
    weights = weights_all[:,1]
    # sum the maps
    heatmap = np.zeros((7,7))
    for idx,weight in enumerate(weights):
        heatmap+=weight*activition_map[:,:,idx]
    # resize the map to proper shape
    heatmap_resized = misc.imresize(heatmap,(resized_l,resized_l),'bilinear') 
    return heatmap_resized


def evaluation(test_label,predictions,plot=True):
    """
    A function that returns plot roc curve
    Args:
        test_labels: labels of the testing data, shape: n*2
        predictions: prediction of the testing data, shape: n*2
    Returns:
        
        
    
    """
    prediction_scaler = np.array(predictions)[:,1]
    prediction_arg = np.argmax(predictions,axis = 1)
    labels_scaler = np.argmax(test_label,axis = 1)
    # calculate accuracy an auc score 
    accuracy = accuracy_score(labels_scaler,prediction_arg )
    auc_score = roc_auc_score(labels_scaler, prediction_scaler)
    recall = recall_score(labels_scaler, prediction_arg)  
    
    # plot the roc_curve if needed
    if plot:
        false_positive_rate, true_positive_rate, thresholds = roc_curve(labels_scaler, prediction_scaler)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        plt.title('Receiver Operating Characteristic')
        plt.plot(false_positive_rate, true_positive_rate, 'b',
        label='AUC = %0.2f'% roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([-0.1,1.2])
        plt.ylim([-0.1,1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
    #print "accuracy is %f"%accuracy
    #print "auc_score is %f"%auc_score
    #print "recall_score is %f"%recall
    return accuracy,auc_score,recall


