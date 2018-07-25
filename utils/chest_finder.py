import cv2
import numpy as np
from scipy import misc


hog = cv2.HOGDescriptor((32, 64), (16, 16), (8, 8), (8, 8), 9, 1, 4, 0, 0.2, 0, 64)

# Find the biggest outline of chest
def find_max_rectangle(rectangles):
    max_rectangle = None
    max_area = 0

    for rectangle in rectangles:
        x, y, width, height = rectangle
        area = width * height

        if area > max_area:
            max_area = area
            max_rectangle = rectangle

    return max_rectangle
	

# Using Haar feature to find right chest
def find_right_lung_haar(image):
    right_lung = cv2.CascadeClassifier("right_lung_haar.xml")
    found = right_lung.detectMultiScale(image, 1.8, 5)
    right_lung_rectangle = find_max_rectangle(found)
    return right_lung_rectangle
	

# Using Haar feature to find left chest	
def find_left_lung_haar(image):
    left_lung = cv2.CascadeClassifier( "left_lung_haar.xml")
    found = left_lung.detectMultiScale(image, 1.8, 5)
    left_lung_rectangle = find_max_rectangle(found)
    return left_lung_rectangle
	
	
# Using HoG feature to find right chest
def find_right_lung_hog(image):
    hog.setSVMDetector(
        np.loadtxt("./utils/right_lung_hog.np", dtype=np.float32))
    found, w = hog.detectMultiScale(image)
    right_lung_rectangle = find_max_rectangle(found)

    return right_lung_rectangle
	

# Using HoG feature to find left chest
def find_left_lung_hog(image):
    hog.setSVMDetector(np.loadtxt( "./utils/left_lung_hog.np", dtype=np.float32))
    found, w = hog.detectMultiScale(image)
    left_lung_rectangle = find_max_rectangle(found)

    return left_lung_rectangle
	

# Using LBP feature to find right chest
def find_right_lung_lbp(image):
    right_lung = cv2.CascadeClassifier( "./utils/right_lung_lbp.xml")
    found = right_lung.detectMultiScale(image, 1.8, 5)
    right_lung_rectangle = find_max_rectangle(found)

    return right_lung_rectangle
	

# Using LBP feature to find left chest	
def find_left_lung_lbp(image):
    left_lung = cv2.CascadeClassifier( "./utils/left_lung_lbp.xml")
    found = left_lung.detectMultiScale(image, 1.8, 5)
    left_lung_rectangle = find_max_rectangle(found)

    return left_lung_rectangle
	

# Combine the lungs found from Haar Feature, HoG feature and get the final chest segmentation
#input arguments:
# image: the input image(2d array)
# padding: control the length enlarged in each dimension for the origined output of algorithm
# top_y_quantile, bottom_y_quantile, x_right_quantile, x_left_quantile: the quantile statistcs for each dimension
# lengh_threhold: control the ratio of lengh(x)/length(y), should be a number less than 1 larger than 0
# size_threhold: control the ratio of area(chest segmented)/area(overall size of image), should be a number less than 1 larger than 0
#
#output:
#image[int(top_y):int(bottom_y), int(x_right):int(x_left_end)]: chest segmented(2d array)
15, 25, 85, 10, 90, 0.6, 0.5
def get_lungs(image, 
              padding=15, 
              top_y_quantile=25, 
              bottom_y_quantile=85,
              x_right_quantile=10, 
              x_left_quantile=90, 
              lengh_threhold=0.6, 
              size_threhold=0.5):
    four_points = np.load('./utils/four_points.npy')
    p1 = np.percentile(four_points[:,0], top_y_quantile)
    p2 = np.percentile(four_points[:,1], bottom_y_quantile)
    p3 = np.percentile(four_points[:,2], x_right_quantile)
    p4 = np.percentile(four_points[:,3], x_left_quantile)
    right_lung = find_right_lung_hog(image)
    left_lung = find_left_lung_hog(image)
    
    if right_lung is not None and left_lung is not None:
        x_right, y_right, width_right, height_right = right_lung
        x_left, y_left, width_left, height_left = left_lung

        if abs(x_right - x_left) < min(width_right, width_left):
            right_lung = find_right_lung_lbp(image)
            left_lung = find_left_lung_lbp(image)

    if right_lung is None:
        right_lung = find_right_lung_haar(image)

    if left_lung is None:
        left_lung = find_left_lung_haar(image)

    if right_lung is None:
        right_lung = find_right_lung_lbp(image)

    if left_lung is None:
        left_lung = find_left_lung_lbp(image)

    if right_lung is None and left_lung is None:
        top_y = p1
        bottom_y = p2
        x_right = p3
        x_left_end = p4
        found_lungs = image[top_y:bottom_y, x_right:x_left_end]
        return top_y, bottom_y, x_right, x_left_end, image[int(top_y):int(bottom_y), int(x_right):int(x_left_end)]
    elif right_lung is None:
        x, y, width, height = left_lung
        spine = width / 5
        right_lung = int(x - width - spine), y, width, height
    elif left_lung is None:
        x, y, width, height = right_lung
        spine = width / 5
        left_lung = int(x + width + spine), y, width, height

    x_right, y_right, width_right, height_right = right_lung
    x_left, y_left, width_left, height_left = left_lung
    
    if width_left < 0.7*width_right:
        width_left = width_right
    
    if width_right < 0.7*width_left:
        width_right = width_left
    
    x_right -= padding
    y_right -= padding
    height_right += padding * 2
    y_left -= padding
    width_left += padding
    height_left += padding * 2
    
    
    if x_right < 0:
        x_right = 0

    if y_right < 0:
        y_right = 0

    if x_left < 0:
        x_left = 0

    if y_left < 0:
        y_left = 0

    if y_right + height_right > image.shape[0]:
        height_right = image.shape[0] - y_right

    if x_left + width_left > image.shape[1]:
        width_left = image.shape[1] - x_left

    if y_left + height_left > image.shape[0]:
        height_left = image.shape[0] - y_left
    
    top_y = min(y_right, y_left)
    bottom_y = max(y_right + height_right, y_left + height_left)
    
    
    if (x_left + width_left - x_right)/float(bottom_y - top_y) <lengh_threhold :
#        print 'lengh less than lengh_threhold'
        if x_right < image.shape[1]*0.3:
            x_left_end = 2*x_left + 2*width_left - x_right
            if x_left_end > p4:
                x_left_end = p4
        else:
            x_right = 2*x_right - x_left + width_left
            x_left_end = x_left + width_left
            if x_right < 0:
                x_right = p3
    else:
        x_left_end = x_left + width_left
    
    if (x_left_end - x_right)*float(bottom_y - top_y) < image.size*size_threhold :
#        print 'area less than lengh_threhold'
        top_y = p1
        bottom_y = p2
        x_right = p3
        x_left_end = p4
		
        
    return image[int(top_y):int(bottom_y), int(x_right):int(x_left_end)]

