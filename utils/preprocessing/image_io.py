import numpy as np
import SimpleITK as sitk
import vtk  # sudo apt-get install python-vtk
from vtk.util import numpy_support
import sklearn
from skimage.io import imread
from skimage.io import imsave
import matplotlib.pyplot as plt
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import warnings
from PIL import ImageEnhance
from PIL import Image as pil_image
# from keras import backend as K
import SimpleITK as sitk


def read_dicom_itk(dir_path,show_shape = False):
    """A function that read dicom series file
        Args: 
            dir_path, a string for dicom path
            show_shape, boolean of whether to show image shape
        Returns:
            Tuple of dicom image spacing and dicom image in numpy format, 
            note dicom file has origin (0,0,0)
    """
    reader = sitk.ImageSeriesReader() 
    dicomfilenames = reader.GetGDCMSeriesFileNames(dir_path) 
    reader.SetFileNames(dicomfilenames) 
    image = reader.Execute() 
    imageRaw = np.flip(sitk.GetArrayFromImage(image),axis=0)
    size = image.GetSize()
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    if show_shape:
        print "Image shape is %s"%(np.array(list(reversed(size))))
    return np.array(list(reversed(spacing))), np.array(list(reversed(origin))),imageRaw


def read_dicom_vtk(dir_path,show_shape = False):
    """A function that read dicom series file
        Args: 
            dir_path, a string for dicom path
            show_shape, boolean of whether to show image shape
        Returns:
            Tuple of dicom image spacing and dicom image in numpy format, 
            note dicom file has origin (0,0,0)
    """
    if not os.path.exists(dir_path):
        raise ValueError('Input path does not exist!')
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(dir_path)
    reader.Update()
    spacing = list(reader.GetPixelSpacing())  # x, y, z
    spacing.reverse()  # z, y, x
    _extent = reader.GetDataExtent()
    ConstPixelDims = [
        _extent[1] - _extent[0] + 1, _extent[3] - _extent[2] + 1,
        _extent[5] - _extent[4] + 1
    ]
    imageData = reader.GetOutput()
    pointData = imageData.GetPointData()
    assert (pointData.GetNumberOfArrays() == 1)
    arrayData = pointData.GetArray(0)
    ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
    # sometimes vtk does not work
    if ArrayDicom.sum() ==0:
        raise ValueError('The output image is empty, try read_dicom_itk!')
    ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')
    ArrayDicom = np.rot90(ArrayDicom, 1)
    ArrayDicom = np.transpose(ArrayDicom, [2, 0, 1])
    if show_shape:
        print "Image shape is %s"%(ArrayDicom.shape,)
    return np.array(spacing), ArrayDicom

def read_mhd(filename,show_shape = False):
    """A function that read mhd file
        Args: 
            filename, a string for image path
        Returns:
            Tuple of image, image spacing and image origin in numpy format, 
    """
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    if show_shape:
        print "Image shape is %s"%(numpyImage.shape,)
    return numpyImage, numpyOrigin, numpySpacing

def read_jpg(filename,show_shape = False):
    """A function that read jpg file
        Args: 
            filename, a string for image path
        Returns:
            image in numpy format, 
    """
    image = imread(filename)
    if len(image.shape) == 2:
        image = image[:,:,np.newaxis]
    if show_shape:
        print "Image shape is %s"%(image.shape,)
    return image

def save_jpgimg(img,save_path):
    """A function that save image to jpg file format
        Args:
            img: numpy array with dimension X*X*3 (or 1)
            save_path: string that defines the image save path
        Return:
            None, image will be saved to defined path
    """
    shape = img.shape
    if len(shape)!=3 or shape[2] not in [1,3]:
        raise TypeError("Input image has dimension %s, which can not be visualized!"%(shape,))
    if shape[2] == 1:
        img = img[:,:,0]
    imsave(save_path,img)
    
def save_npimg(img,save_path):
    """A function that save image to jpg file format
        Args:
            img: numpy array with dimension X*X*3 (or 1)
            save_path: string that defines the image save path
        Return:
            None, image will be saved to defined path
    """
    
    np.save(save_path,img)

def show_image(img,figsize = (8,8)):
    """A function that visualize image
        Args:
            img: numpy array with dimension X*X*3 (or 1)
            figsize: tuple with dimension 1*2 that defines the image visualization size
        Return:
            None, image will be visualized
    """
    fig = plt.figure(figsize=figsize)
    if len(img.shape) == 2:
        img = img[:,:,np.newaxis]
    shape = img.shape
    if len(shape)!=3 or shape[2] not in [1,3]:
        raise TypeError("Input image has dimension %s, which can not be visualized!"%(shape,))
    if shape[2]==1:
        plt.imshow(img[:,:,0],cmap="bone")
    else:
        plt.imshow(img)
    plt.show()