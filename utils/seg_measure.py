import os
import cv2
import numpy as np
import copy
import math
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import convex_hull_image
from skimage.morphology import disk, binary_erosion, binary_closing
from skimage.measure import label, regionprops
from skimage.filters import roberts
from skimage import measure
from skimage.segmentation import clear_border
import skimage
from scipy import ndimage as ndi
from scipy.ndimage.interpolation import zoom
from attached import get_chest_boundary, boundary_attached, normalize
from scipy.misc import imresize
from multiprocessing import cpu_count, Pool
from skimage.draw import line_aa
from scipy.misc import imfilter


def cutnodule(nodule, parameter):
    row, col = nodule.shape
    seed = [row/2, col/2]
    average = np.average(nodule)
    var = np.std(nodule)
    threshold = average + parameter*var
    im = nodule > threshold
    rev = nodule < threshold
    #plt.imshow(rev, cmap = 'gray')
    #plt.show()
    extend = np.ones(shape = [row+10, col+10])
    extend[0:5, :] = 0
    extend[extend.shape[0]-5:extend.shape[0], :] = 0
    extend[:, 0:5] = 0
    extend[:, extend.shape[0]-5:extend.shape[0]] = 0
    extend[5:extend.shape[0]-5, 5:extend.shape[0]-5] = extend[5:extend.shape[0]-5, 5:extend.shape[0]-5]*rev
    #rev_labels = label(extend, connectivity = 1)
    filled = np.zeros(extend.shape)
    #plt.imshow(extend, cmap = 'gray')
    #plt.show()
    
    selem = disk(10)
    #for i in range(1, np.max(rev_labels) + 1):
    curr_region = np.zeros(filled.shape)
    curr_region[extend == True] = 1
    #curr_region = binary_closing(curr_region, selem)
    curr_region = binary_fill_holes(curr_region)
    curr_region = convex_hull_image(curr_region)
    filled[curr_region == 1] = 1
    final_filled = filled[5:extend.shape[0]-5, 5:extend.shape[0]-5]
    #plt.imshow(final_filled, cmap = 'gray')
    #plt.show()
  
    res = np.abs(final_filled - im)
    final = res == 0
    final_labels = label(final, connectivity = 1)
    #plt.imshow(final_labels, cmap = 'gray')
    #plt.show()
    for region in regionprops(final_labels):
        if region.centroid[0] < 0.2*row \
        or region.centroid[1] > 0.8*col \
        or region.centroid[1] < 0.2*col \
        or region.centroid[0] > 0.8*row \
        or region.eccentricity > 1 or region.area < 5:
            for coord in region.coords:
                final_labels[coord[0], coord[1]] = 0
        else:
            [min_row, min_col, max_row, max_col] = region.bbox
    
    return final_labels


def region_growing_2d(img, seed, region_threshold):
    #Parameters for region growing
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    region_size = 1
    intensity_difference = 0
    seg_list = []
    test_list = []
    test_intensity_list = []
    seg_list.append(seed)
    height, width = img.shape
    sum_pixel = img[seed]
    count_pixel = 1
    for i in range(4):
        y_new = seed[0] + neighbors[i][0]
        x_new = seed[1] + neighbors[i][1]

        check_inside = (x_new >= 0) & (y_new >= 0) & (x_new < width) & (y_new < height)

        if (y_new, x_new) in seg_list:
            continue

        if check_inside:
            test_list.append((y_new, x_new))

    region_mean = img[seed]
    

    image_size = 30 * 30

    segmented_img = np.zeros((height, width), np.float32)
    segmented_img[seed] = 1
    
    while (region_size < image_size):
        temp_list = []
        for j in range(len(test_list)):
            intensity = img[test_list[j]]
            distance = intensity-region_mean
            if intensity >= region_threshold:
                temp_list.append(test_list[j])
                sum_pixel += intensity
                count_pixel+=1
                region_mean = sum_pixel/float(count_pixel)
        if len(temp_list) == 0:
            break
        else:
            seg_list = seg_list+temp_list
            seg_list = list(set(seg_list))
            
        test_list = [] 
        for i in range(len(seg_list)):
            for j in range(4):
                y_new = seg_list[i][0] + neighbors[j][0]
                x_new = seg_list[i][1] + neighbors[j][1]

                check_inside = (x_new >= 0) & (y_new >= 0) & (x_new < width) & (y_new < height)

                if check_inside:
                    test_list.append((y_new, x_new))

            
        region_size = len(set(seg_list))
        test_list = list(set(test_list)-set(seg_list))
        #print len(test_list)
    
    for i in range(len(seg_list)):
        segmented_img[seg_list[i]] = 1
    return segmented_img


def final_cut(nodule, margin):
    #plt.imshow(nodule, cmap = 'gray')
    #plt.show()
    blur = cv2.bilateralFilter(nodule.astype(np.float32),9,130,130)
    th_temp = skimage.filters.threshold_otsu(blur)
    #th_temp = 0.5*blur[margin, margin]
    grow = region_growing_2d(blur, (margin, margin), th_temp)
    output = cutnodule(blur, parameter = 0.3)
    final_res = output*grow
    #make sure there is only one nodule
    if len(regionprops(label(final_res))) > 1:
        maxi_prev = regionprops(label(final_res))[0]
        for region in regionprops(label(final_res)):
            if region.area < maxi_prev.area:
                for coord in region.coords:
                    final_res[coord[0], coord[1]] = 0
            elif region.area > maxi_prev.area:
                for coord in maxi_prev.coords:
                    final_res[coord[0], coord[1]] = 0
                maxi_prev = region
            else:
                continue
    return final_res


def drawline(theta, row_c, col_c, row, col):
    radians = math.radians(theta)
    
    
    tang = math.tan(radians)
    row_t = []
    col_t = []
    if theta == 45:
        row_t.append(0)
        row_t.append(row-1)
        col_t.append(col-1)
        col_t.append(0)
    if theta == 135:
        row_t.append(0)
        row_t.append(row-1)
        col_t.append(0)
        col_t.append(col-1)
    if theta is 90:
        row_t.append(0)
        row_t.append(row-1)
        col_t.append(col_c)
        col_t.append(col_c)
    elif theta is 0:
        row_t.append(row_c)
        row_t.append(row_c)
        col_t.append(0)
        col_t.append(col-1)
    else:
        row_V = tang*(col-col_c)
        col_V = (float(row_c))/float(tang)
        if abs(row_V) > row_c:
            row_t.append(0)
            row_t.append(row-1)
            col_t.append(col_c + int(col_V))
            col_t.append(col_c - int(col_V))
        elif abs(col_V) > col - col_c:
            col_t.append(0)
            col_t.append(col-1)
            row_t.append(row_c + int(row_V))
            row_t.append(row_c - int(row_V))

    rr, cc, val = line_aa(row_t[0], col_t[0], row_t[1], col_t[1])
    return rr, cc


#a function to calculate the lenght
def calculate_length(nodule, theta, diameter, delta_row, delta_col, row, col, spc):
    #get the all pixels of this nodule
    coord = nodule[0].coords
    center_row, center_col = int(nodule[0].centroid[0]), int(nodule[0].centroid[1])
    len_pos = []
    #calculate the length in different situation
    if theta is 0:
        min_col = col
        max_col = 0
        for i in range(coord.shape[0]):
            pos = coord[i]
            if diameter[pos[0], pos[1]] == 0:
                if pos[1] < min_col:
                    min_col = pos[1]
                if pos[1] > max_col:
                    max_col = pos[1]
        length = float(max_col - min_col)*spc[2]
        len_pos.append([center_row+delta_row, min_col+delta_col])
        len_pos.append([center_row+delta_row, max_col+delta_col])
        
    if theta is 90:
        min_row = row
        max_row = 0
        for i in range(coord.shape[0]):
            pos = coord[i]
            if diameter[pos[0], pos[1]] == 0:
                if pos[0] < min_row:
                    min_row = pos[0]
                if pos[0] > max_row:
                    max_row = pos[0]
        length = float(max_row - min_row)*spc[1]
        len_pos.append([min_row+delta_row, center_col+delta_col])
        len_pos.append([max_row+delta_row, center_col+delta_col])
    else:
        min_row = row
        max_row = 0
        min_col = col
        max_col = 0
        for i in range(coord.shape[0]):
            pos = coord[i]
            if diameter[pos[0], pos[1]] == 0:
                if pos[0] < min_col:
                    min_row = pos[0]
                    min_col = pos[1]
                if pos[0] > max_col:
                    max_row = pos[0]
                    max_col = pos[1]
        height = float(max_col - min_col)*spc[2]
        width = float(max_row - min_row)*spc[1]
        length = math.sqrt(height*height + width*width)
        len_pos.append([min_row+delta_row, min_col+delta_col])
        len_pos.append([max_row+delta_row, max_col+delta_col])
    return length, int(length/spc[2]), len_pos


#input is a whole ct with nodule which is already labeled
def diameterandvolume(cutnodule, spc, margin, num_layer):
    volume = 0
    max_len = 0
    max_degree = 0
    max_len_pixel = 0
    min_len = 0
    min_len_pixel = 0
    min_len_pos = []
    max_len_pos = []
    max_layer = 0
    
    #take out each layer of this nodule and calculate the diameter
    for i in range(num_layer[0], num_layer[1]):
        nodule_temp = cutnodule[i]
        #get the center point of the nodule & cut the nodule
        nodule_label = regionprops(label(nodule_temp))
        if len(nodule_label) is 1:
            row_c = int(nodule_label[0].centroid[0])
            col_c = int(nodule_label[0].centroid[1])
            nodule_cut = nodule_temp[row_c-margin:row_c+margin, col_c-margin:col_c+margin]
            delta_row = row_c - margin
            delta_col = col_c - margin
            nodule = regionprops(label(nodule_cut))
            volume += nodule[0].area
            row = margin*2
            col = margin*2

            #now calculate the diameter
            for theta in range(0, 180, 5):
                #print theta
                line_img = np.zeros(shape = [row, col])
                rr, cc = drawline(theta, margin, margin, row, col)
                try:
                    line_img[rr, cc] = np.max(label(nodule_cut))
                except Exception, err:
                    continue
                diameter = abs(line_img - label(nodule_cut))
                #plt.imshow(diameter, cmap = 'gray')
                #plt.show()

                length, length_pixel, len_pos = calculate_length(nodule, theta, diameter, delta_row, delta_col, row, col, spc)
                if length > max_len and length < row:
                    max_len = length
                    max_len_pixel = length_pixel
                    max_degree = theta
                    max_len_pos = len_pos
                    max_layer = i


                    #calculate the short diameter
                    line_img = np.zeros(shape = [row, col])
                    if max_degree < 90:
                        theta = max_degree + 90
                    else:
                        theta = max_degree - 90
                    rr, cc = drawline(theta, margin, margin, row, col)
                    try:
                        line_img[rr, cc] = np.max(label(nodule_cut))
                        diameter_short = abs(line_img - label(nodule_cut))
                        min_len, min_len_pixel, min_len_pos = calculate_length(nodule, theta, diameter_short, delta_row, delta_col, row, col, spc)
                    except Exception, err:
                        min_len = 0
                        
        else:
            continue
        
    return max_len, max_len_pixel, min_len, min_len_pixel, max_len_pos, min_len_pos, volume, max_layer


def mask_pos(cutnodule, max_layer):
    results = []
    #label the nodule first
    nodule = cutnodule[max_layer]
    [min_row, min_col, max_row, max_col] = regionprops(label(nodule))[0].bbox

    filled_edge = imfilter(nodule.astype(np.float64), 'find_edges') / 255
    for row in range(0, filled_edge.shape[0]):
        for col in range(0, filled_edge.shape[1]):
            if filled_edge[row, col] == 1:
                results.append([row, col])
    boundingbox = [[min_row,min_col], [min_row,max_col], [max_row, max_col], [max_row, min_col]]
    return results, boundingbox

def nodule_segment(imgs, area_size, seed, margin):
    #to see which layer is the start and which is end
    z = int(seed[0])
    y = int(seed[1])
    x = int(seed[2])
    min_zz = 0
    max_zz = 0
    center_y = y
    center_x = x
    delta_y = 0
    delta_x = 0
    for i in range(-5,0,1):
        temp = imgs[z+i, center_y+delta_y-margin:center_y+delta_y+margin, center_x+delta_x-margin:center_x+delta_x+margin]
        final_res = final_cut(temp, margin)
        if len(regionprops(label(final_res))) > 0:
            delta_y = center_y + delta_y - margin
            delta_x = center_x + delta_x - margin
            area_temp = regionprops(label(final_res))[0].area
            center_y, center_x = int(regionprops(label(final_res))[0].centroid[0]), int(regionprops(label(final_res))[0].centroid[1])
            
            if area_temp > 0.4*area_size and area_temp < 1.5*area_size:
                min_zz = z+i
                break

    center_y = y
    center_x = x
    delta_y = 0
    delta_x = 0
    for i in range(5,0,-1):
        temp = imgs[z+i, center_y+delta_y-margin:center_y+delta_y+margin, center_x+delta_x-margin:center_x+delta_x+margin]
        final_res = final_cut(temp, margin)
        if len(regionprops(label(final_res))) > 0:
            delta_y = center_y + delta_y - margin
            delta_x = center_x + delta_x - margin
            area_temp = regionprops(label(final_res))[0].area
            center_y, center_x = int(regionprops(label(final_res))[0].centroid[0]), int(regionprops(label(final_res))[0].centroid[1])
            
            if area_temp > 0.4*area_size and area_temp < 1.5*area_size:
                max_zz = z+i
                break

    if min_zz is 0:
        min_zz = z
    if max_zz is 0:
        max_zz = z

    nodule_3d = imgs[min_zz:max_zz+1, y-margin:y+margin, x-margin:x+margin]
    num_layer = (min_zz, max_zz+1)
    result = np.zeros(imgs.shape)
    summ = 0
    count = 0
    for i in range(nodule_3d.shape[0]):
        temp_img = final_cut(nodule_3d[i], margin)
        result[min_zz+i, y-margin:y+margin, x-margin:x+margin] = temp_img
        if len(regionprops(label(temp_img))) > 0:
            area_temp = regionprops(label(temp_img))[0].area
            mask_temp = (temp_img*imgs[min_zz+i, y-margin:y+margin, x-margin:x+margin])/np.max(temp_img)
            summ += np.sum(mask_temp)
            count += area_temp
    return result, num_layer, summ*1.0/count


def last_step(imgs, seed, spc, ori):
    margin = 5
    z, row, col = seed[0], seed[1], seed[2]
    #print z, row, col
    #print imgs.shape
    nodule = imgs[z, row-margin:row+margin, col-margin:col+margin]
    #plt.imshow(nodule, cmap = 'gray')
    #plt.show()
    nodule_seg_2d = final_cut(nodule, margin)
    if len(regionprops(label(nodule_seg_2d))) > 0 and regionprops(label(nodule_seg_2d))[0].area > 5:
        area_size = regionprops(label(nodule_seg_2d))[0].area
        nodule_seg_3d, num_layer, density = nodule_segment(imgs, area_size, seed, margin)
        
        print 'segmentation done!'
        max_len, max_len_pixel, min_len, min_len_pixel, max_len_pos, min_len_pos, volume, max_layer = diameterandvolume(nodule_seg_3d, spc, margin, num_layer)
        #maskpositions, boundingbox = mask_pos(nodule_seg_3d, max_layer)
        return [max_len, min_len, density]
    else:
        return [0, 0, 0]
        #print 'nothing is segmented!'
        
    
    
    


