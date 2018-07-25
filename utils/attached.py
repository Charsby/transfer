import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import time
import cv2
from tqdm import tqdm
from skimage.morphology import disk, binary_erosion, binary_closing
from skimage.measure import label, regionprops
from skimage.filters import roberts
from skimage import measure
from skimage.util import img_as_float
from skimage.segmentation import clear_border
import scipy.misc as misc
from scipy import ndimage as ndi
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.morphology import binary_dilation
from itertools import combinations_with_replacement
from sklearn import mixture,cluster
from scipy.ndimage.morphology import binary_fill_holes
from skimage import feature



def ReadDICOMFolder(folderName):
    '''Load DCM series from a folder.
    '''
    reader = sitk.ImageSeriesReader()
    seriesIDs = reader.GetGDCMSeriesIDs(folderName)
    for idx in seriesIDs:
        try:
            dicomfilenames = reader.GetGDCMSeriesFileNames(folderName,idx)
            #dicomfilenames = reader.GetGDCMSeriesFileNames(folderName) 
            reader.SetFileNames(dicomfilenames)
            image = reader.Execute()
            imageRaw = sitk.GetArrayFromImage(image)
            size = image.GetSize()
            origin = image.GetOrigin()
            spacing = image.GetSpacing()
            direction = image.GetDirection()
            imageRaw = np.flip(imageRaw, 0)
        except:
            pass
        if len(imageRaw) >= 200:
            break
    return imageRaw, spacing


def get_chest_boundary(im, plot=False):
    
    size = im.shape[1]
    if plot == True:
        f, plots = plt.subplots(6, 1, figsize=(5, 30))
    binary = im < -320
    
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone)

    cleared = clear_border(binary)
    temp_label = label(cleared)
    for region in regionprops(temp_label):
        if region.area < 300:
            for coordinates in region.coords:
                temp_label[coordinates[0], coordinates[1]] = 0
    cleared = temp_label > 0
    
    label_img = label(cleared)
    for region in regionprops(label_img):
        if region.eccentricity > 0.99 \
                or region.centroid[0] > 0.90 * size \
                or region.centroid[0] < 0.12 * size \
                or region.centroid[1] > 0.88 * size \
                or region.centroid[1] < 0.10 * size \
                or (region.centroid[1] > 0.46 * size and region.centroid[1] < 0.54 * size and region.centroid[
                    0] > 0.75 * size) \
                or (region.centroid[0] < 0.2 * size and region.centroid[1] < 0.2 * size) \
                or (region.centroid[0] < 0.2 * size and region.centroid[1] > 0.8 * size) \
                or (region.centroid[0] > 0.8 * size and region.centroid[1] < 0.2 * size) \
                or (region.centroid[0] > 0.8 * size and region.centroid[1] > 0.8 * size):
            for coordinates in region.coords:
                label_img[coordinates[0], coordinates[1]] = 0
    
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(label_img, cmap=plt.cm.bone)  
        
    region_n = np.max(label_img)
    selem = disk(10)
    filled = np.zeros(cleared.shape, np.uint8)
    for i in range(1, region_n+1):
        curr_region = np.zeros(cleared.shape, np.uint8)
        curr_region[label_img == i] = 1
        curr_region = binary_closing(curr_region, selem)
        curr_region = binary_fill_holes(curr_region)
        filled[curr_region == 1] = 1
    
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(filled, cmap=plt.cm.bone)
        
    filled_edge = misc.imfilter(filled.astype(np.float64), 'find_edges') / 255
    
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(filled_edge, cmap=plt.cm.bone)

    return filled_edge


def boundary_attached(ct, spacing, x, y, z, diameter, plot=False, chest_b_thickness=3):
    '''input ct should be a 3d ndarray'''
    
    centre_slice = ct[z]
    if plot==True:
        display = np.expand_dims((normalize(centre_slice) * 255).astype(np.uint8),2)
        display = np.concatenate((display, display, display), axis=2)
        display = cv2.circle(display, (x,y), 2, (255,0,0), -1)
        plt.figure(figsize=(5,5))
        plt.imshow(display)
        plt.show() 
    
    peri_d = int(diameter / spacing[1] / 2)
    chest_boundary_mask = get_chest_boundary(centre_slice).astype(np.bool_)
    chest_boundary_mask = ndi.binary_dilation(chest_boundary_mask, iterations=chest_b_thickness)
    nodule_area = np.zeros(centre_slice.shape).astype(np.uint8)
    nodule_area = cv2.circle(nodule_area, (x,y), peri_d, 255, -1).astype(np.bool_)
    intersect = np.sum(np.logical_and(chest_boundary_mask, nodule_area))
  
    if intersect > 0:            
        return True
    else:
        return False
    
def normalize(x):
    return (x - np.min(x)) * 1. / (np.max(x) - np.min(x))