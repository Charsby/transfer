
from scipy.ndimage.interpolation import rotate
import pandas as pd
import numpy as np
import cv2
from scipy import misc
import sklearn
import sklearn.preprocessing as preprocessing
import skimage.transform as transform
import SimpleITK as sitk
import matplotlib.pyplot  as plt
from chest_finder import get_lungs
from tqdm import tqdm


def data_generator(img_paths,
                   labels,
                   batch_size = 1,
                 RESIZE_W=299,
                 RESIZE_H=299,
                   seg = False,
                  augment = False):
    '''
    A generator that convert image path to appropriate image
    
    Args:
        img_paths: a list for image paths. Supported image file type is Dicom and JPG/PNG
        labels: a numpy array for image corresponding labels
        batch_size: batch size
        RESIZE_W: resized width for output images
        RESIZE_H: resized height for output images 
        augment: whether to augment the images
    
    
    Returns:
        Batched data and its corresponding batched labels
    '''
    # sanity check
    assert len(img_paths)==len(labels),'the length of image paths and labels must be same'
    while 1:
        batch_data=[]
        batch_labels=[]

        while len(batch_data)<batch_size:
            # randomly select a image
            idx = np.random.randint(len(img_paths))
            img_path = img_paths[idx]
            label = labels[idx]
            # use lung seg image
            if seg:
                if not "NLMCXR" in img_path:
                    img_path = img_path.replace('jpg','chest_jpg')
                    img_path = img_path[:-9]+'jpg'
                    
            # try different encode pattern
            try:
                image_single = misc.imread(img_path)   
            except:
                try:
                    sitk_image = sitk.ReadImage(img_path)
                    img = sitk.GetArrayFromImage(sitk_image)
                    image_single = img[0]
                except:
                    try:
                        sitk_image = sitk.ReadImage(img_path.encode("utf-8"))
                        img = sitk.GetArrayFromImage(sitk_image)
                        image_single = img[0]
                    except:
#                        print "image can not be loaded for file path %s"%img_path
                        continue
            # some images may contain wield channels
            try :
                if (len(image_single.shape)>2): 
                    image_single = image_single[:,:,0]
            except:
                pass
            

            
            # augmentation if necessary
            if augment:
                if label[1] == 1:
                    # randomly flip the image
                    if np.random.choice([0,1]):
                        image_single = np.fliplr(image_single)
                    # randomly crop out the center of image 
                    if np.random.choice([0,1]):
                        interval = int(np.random.uniform(0,0.1)*image_single.shape[0])

                        image_single = image_single[interval:image_single.shape[0]-interval,
                                                   interval:image_single.shape[1]-interval]
                    # randomly rorate the image
                    if np.random.choice([0,1]):
                        theta = np.random.uniform(-8, 8)
                        image_single = rotate(image_single,theta)
                    #if np.random.choice([0,1]):
                    #    image_single = cv2.equalizeHist(image_single)
                                  
            # reverse the image if necessary
            if (image_single[:5,:5].mean()+
                image_single[-5:,-5:].mean())/2>image_single.mean():
                image_single = (image_single.max()-image_single)

            # resize the image to fit the net
            resized_image = transform.resize(image_single,(RESIZE_W,RESIZE_H,1),preserve_range=True)
#            plt.imshow(resized_image[:,:,0],cmap="bone")
#            plt.show()
            # normalize the image
            norm_img = (resized_image-resized_image.min())/float((resized_image.max()-resized_image.min()))
            norm_img = (norm_img-0.5)*2
            # stack to 3 channel
            image = np.concatenate((norm_img ,
                                    norm_img ,
                                    norm_img ),axis=2)
            batch_data.append(image)
            batch_labels.append(label)

        yield np.array(batch_data),np.array(batch_labels)

def test_generator(img_paths, RESIZE_W, RESIZE_H):
    test_images = []
    for img_path in img_paths:
        try:
            image_single = misc.imread(img_path)   
        except:
            try:
                sitk_image = sitk.ReadImage(img_path)
                img = sitk.GetArrayFromImage(sitk_image)
                image_single = img[0]
            except:
                try:
                    sitk_image = sitk.ReadImage(img_path.encode("utf-8"))
                    img = sitk.GetArrayFromImage(sitk_image)
                    image_single = img[0]
                except:
                    print img_path
                    continue
        # some images may contain wield channels
        
        try :
            if (len(image_single.shape)>2): 
                image_single  = image_single[:,:,0]
        except:
            pass              
        # resize the image to fit the net
        resized_image = transform.resize(image_single,(RESIZE_W,RESIZE_H,1),preserve_range=True)

        # normalize the image
        norm_img = (resized_image-resized_image.min())/float((resized_image.max()-resized_image.min()))
        norm_img = (norm_img-0.5)*2
        # stack to 3 channel
        image = np.concatenate((norm_img ,
                                norm_img ,
                                norm_img ),axis=2)
        #image = np.expand_dims(image, 3)
        test_images.append(image)
    return np.array(test_images)
        