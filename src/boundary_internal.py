import os
import sys
import numpy as np
import scipy.ndimage as nd
import keras
from keras import backend as K
from keras.models import load_model

def get_bi_classifier(MODEL_PATH):
    K.clear_session()
    K.set_learning_phase(False)
    model = load_model(MODEL_PATH)
    return model
    

def bi_classification(model, image, centroid, diameter, spacing):
    return [0]*5