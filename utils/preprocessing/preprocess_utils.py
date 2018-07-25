from multiprocessing import Pool, cpu_count
import numpy as np
import SimpleITK as sitk
from skimage.morphology import disk, binary_erosion, binary_closing
from skimage.measure import label, regionprops
from skimage.filters import roberts
from skimage import measure
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
from scipy.ndimage.interpolation import zoom
import vtk  # sudo apt-get install python-vtk
from vtk.util import numpy_support
import warnings
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes
import os


def load_itk(filename):
    try:
        itkimage = sitk.ReadImage(filename)
        numpyImage = sitk.GetArrayFromImage(itkimage)
        numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
        numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
        numpyImage = np.flip(numpyImage, 0)
        uid = os.path.split(filename)[-1]
        if '.mhd' in filename:
            uid = uid.split('.mhd')[0]
        else:
            uid = uid.split('.nii')[0]
        return numpyImage, numpySpacing, numpyOrigin, uid
    except:
        raise Exception('Fail to load the CT file ' + filename)

def ReadDICOMFolder(folderName, input_uid=None):
    '''A nearly perfect DCM reader!'''
    reader = sitk.ImageSeriesReader()
    out_uid = ''
    out_image = None
    max_slice_num = 0

    # if the uid is not given, iterate all the available uids
    try:
        uids = [input_uid] if input_uid!=None else reader.GetGDCMSeriesIDs(folderName)
    except TypeError:
        folderName = folderName.encode('utf-8')
        uids = [input_uid] if input_uid!=None else reader.GetGDCMSeriesIDs(folderName)

    for uid in uids:
        try:
            dicomfilenames = reader.GetGDCMSeriesFileNames(folderName,uid)
            reader.SetFileNames(dicomfilenames)
            image = reader.Execute()
            size = image.GetSize()
            if size[0] == size[1] and size[2]!=1: # exclude xray
                slice_num = size[2]
                if slice_num > max_slice_num:
                    out_image = image
                    out_uid = uid
                    max_slice_num = slice_num
        except:
            pass
    if out_image != None:
        imageRaw = sitk.GetArrayFromImage(out_image)
        imageRaw = np.flip(imageRaw, 0)
        spacing = list(out_image.GetSpacing())
        spacing.reverse()
        origin = list(out_image.GetOrigin())
        origin.reverse()
        return imageRaw, np.array(spacing), np.array(origin),uid
    else:
        raise Exception('Fail to load the dcm folder '+folderName)


def load_CT(folderName):
    if os.path.isdir(folderName):
        imgs, spc, ori, uid = ReadDICOMFolder(folderName)
    else:
        imgs, spc, ori, uid = load_itk(folderName)
#     print imgs.shape
    return imgs, spc, ori, uid

def get_segmented_lungs_mask(im, idx, plot=False):
    size = im.shape[1]

    if plot == True:
        f, plots = plt.subplots(5, 1, figsize=(5, 25))
    binary = im < -320

    #if plot == True:
        #plots[0].axis('off')
        #plots[0].imshow(im, cmap=plt.cm.bone)
        
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(binary, cmap=plt.cm.bone)

    cleared = clear_border(binary)
    temp_label = label(cleared)
    for region in regionprops(temp_label):
        if region.area < 300:
            for coordinates in region.coords:
                temp_label[coordinates[0], coordinates[1]] = 0
    cleared = temp_label > 0
    cleared = binary_fill_holes(cleared)
    
    
    label_img = label(cleared)
    for region in regionprops(label_img):
        if region.eccentricity > 0.99 \
                or region.centroid[0] > 0.86 * size \
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
        plots[2].axis('off')
        plots[2].imshow(label_img, cmap=plt.cm.bone)
            
    regions_ori = regionprops(label_img)
    regions_ori = sorted(regions_ori, key=lambda x:x.area, reverse=True)
    if len(regions_ori)>0 and float(regions_ori[0].area) / float(size*size) > float(150000)/float(512*512):
        #print 'only one lung'
        count = 0
        regions_temp = regionprops(label_img)
        regions_temp = sorted(regions_temp, key=lambda x:x.area, reverse=True)
        erode_label_img = label_img
        
        while(regions_temp[0].area>regions_ori[0].area*0.7):
            erode_label_img = ndi.binary_erosion(erode_label_img>0, iterations=2)
            erode_label_img = label(erode_label_img)
            regions_temp = regionprops(erode_label_img)
            regions_temp = sorted(regions_temp, key=lambda x:x.area, reverse = True)
            count += 1
        region_n = np.max(erode_label_img)
        new_label_img = np.zeros(erode_label_img.shape, np.uint8)
        for i in range(1, region_n+1):
            curr_region = np.zeros(erode_label_img.shape, np.uint8)
            curr_region[erode_label_img == i] = 1
            curr_region = ndi.binary_dilation(curr_region, iterations=count*2)
            new_label_img[curr_region == 1] = i
        label_img = new_label_img
     
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(new_label_img, cmap=plt.cm.bone)       
    
    region_n = np.max(label_img)
    selem = disk(20)
    filled = np.zeros(cleared.shape, np.uint8)
    for i in range(1, region_n+1):
        curr_region = np.zeros(cleared.shape, np.uint8)
        curr_region[label_img == i] = 1
        curr_region = binary_closing(curr_region, selem)
        filled[curr_region == 1] = 1
    
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(filled, cmap=plt.cm.bone)
        
    return filled, idx


def mask_extractor(imgs, uid, area_th=0.1, min_area=1000, debug=False):
    
    #imgs, spc, ori, uid = load_CT(path)
    size = imgs.shape[1]
    if debug:
        print 'load ' + uid + ' done!'
    #assert spc[0] < 6, "slice thickness must < 6mm!"
    
    imgs = denoising(imgs)
    imgs = vesselfiltering(imgs)
    
    if debug:
        print 'filtering done! ' + uid
    # binarize each frame
    results = []
    pool = Pool(cpu_count())
    for i in range(imgs.shape[0]):
        result = pool.apply_async(get_segmented_lungs_mask,(
            imgs[i],
            i
        ))
        
        results.append(result)
    
    pool.close()
    pool.join()
    
    # res is an ApplyResult Object, use .get() to obtain data
    results = [res.get() for res in results]
    im_mask = np.zeros_like(imgs, dtype=np.bool)
    for (msk, i) in results:
        im_mask[i] = msk

    if debug:
        print 'mask ready! ' + uid
        
    while True:
        label = measure.label(im_mask)
        properties = measure.regionprops(label)
        total = len(properties)
        assert len(properties) > 0, "empty image"
        properties.sort(key=lambda x: x.area, reverse=True)
        summ = 0
        for i in range(0, total):
            summ += properties[i].area
        #print total, summ
        # keep the largest connected region
        # keep any region that is larger than area_th
        valid_label = set()
        for index, prop in enumerate(properties):
            if index == 0:
                valid_label.add(prop.label)
            else:
                if float(prop.area)/float(summ) > area_th:
                    valid_label.add(prop.label)
                else:
                    break
        current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
        
        zz, yy, xx = np.where(current_bw)
        cut_length = np.max(xx) - np.min(xx)
        if cut_length > 0.5*size:
            break
        else:
            area_th -= 0.02
            if area_th < 0:
                break
            else:
                continue
    
    if debug:
        print 'loop done! ' + uid
    
    # delete a few starting and ending frames
    for i in range(current_bw.shape[0]):
        if current_bw[i].sum() < min_area:
            current_bw[i] = 0
        else:
            break
    for i in reversed(range(current_bw.shape[0])):
        if current_bw[i].sum() < min_area:
            current_bw[i] = 0
        else:
            break
    
    '''
    if debug:
        for i in range(current_bw.shape[0]):
            #print 'Wrong!'
            current_bw[i].sum()
    '''      
    # 3d-dilation for making sure that marginal nodules are involved
    mask_2 = current_bw
    
    if debug:
        print 'mask done! ' + uid
        #plt.imshow(mask_2[mask_2.shape[0] / 2], cmap="bone")
    return imgs, mask_2


def lumTrans(img):
    lungwin = np.array([-1000., 600.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg


def resample(imgs, spacing, new_spacing, order=2):
    if len(imgs.shape) == 3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imgs = zoom(imgs, resize_factor, mode='nearest', order=order)
        return imgs, true_spacing
    elif len(imgs.shape) == 4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:, :, :, i]
            newslice, true_spacing = resample(slice, spacing, new_spacing)
            newimg.append(newslice)
        newimg = np.transpose(np.array(newimg), [1, 2, 3, 0])
        return newimg, true_spacing
    else:
        raise ValueError('wrong shape')


def mask_processing(im, mask, spacing):
    resolution = np.array([1, 1, 1])
    newshape = np.round(np.array(mask.shape) * spacing / resolution)
    xx, yy, zz = np.where(mask)
    box = np.array([[np.min(xx), np.max(xx)], 
                    [np.min(yy), np.max(yy)],
                    [np.min(zz), np.max(zz)]])
    box = box * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
    box = np.floor(box).astype('int')
    margin = 12
    extendbox = np.vstack([
        np.max([[0, 0, 0], box[:, 0] - margin], 0),
        np.min([newshape, box[:, 1] + 2 * margin], axis=0).T
    ]).T
    extendbox = extendbox.astype('int')

    bone_thresh = 210
    pad_value = 170

    sliceim = lumTrans(im)
    sliceim = sliceim * mask + pad_value * (1 - mask).astype('uint8')
    bones = sliceim * mask > bone_thresh
    sliceim[bones] = pad_value
    #print "sliceim", sliceim.shape
    #sliceim1 = sliceim[extendbox[0, 0]:extendbox[0, 1], extendbox[1, 0]:
     #                   extendbox[1, 1], extendbox[2, 0]:extendbox[2, 1]]
    #print "sliceim1", sliceim1.shape
    #sliceim2, true_spacing = resample(sliceim1, spacing, resolution, order=1)
    #print "sliceim2", sliceim2.shape

    sliceim1, true_spacing = resample(sliceim, spacing, resolution, order=1)
    sliceim2 = sliceim1[extendbox[0, 0]:extendbox[0, 1], extendbox[1, 0]:
                        extendbox[1, 1], extendbox[2, 0]:extendbox[2, 1]]

    sliceim = sliceim2[np.newaxis, ...]
    return sliceim, extendbox, true_spacing

def denoising(img):
    return img

def vesselfiltering(img):
    return img