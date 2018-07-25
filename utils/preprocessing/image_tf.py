
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math
import numpy as np
def parse_function_jpg(filename,resize_shape = (300,300)):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string,channels=3)
    image_resized = tf.image.resize_images(image_decoded, resize_shape)
    return image_resized

def parse_function_png(filename,resize_shape = (300,300)):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string,channels=4)
    image_resized = tf.image.resize_images(image_decoded, resize_shape)
    return image_resized


def random_rotate(image,rotate_ang_high):
    """Randomly rotate the image by 4 multiples of 90.
    
    Args : 
        image: A 3-D image `Tensor`.
        rotate_ang_high: the upper bound of rotation in degree
    
    Returns:
        A randomly rotated and flipped image `Tensor` from the provided rotation angle list
        
    """
    # random rotate
    theta = math.pi/180*np.random.uniform(-rotate_ang_high,rotate_ang_high)
    rotated_image = tf.contrib.image.rotate(image,theta)
    
    return rotated_image 

def random_rotate_fixed(image):
    """Randomly rotate the image by 4 multiples of 90.
    
    Args : 
        image: A 3-D image `Tensor`.
        rotate_ang_high: the upper bound of rotation in degree
    
    Returns:
        A randomly rotated and flipped image `Tensor` from the provided rotation angle list
        
    """
    # random rotate
    elems = tf.convert_to_tensor([math.pi/180*0,math.pi/180*90,math.pi/180*180,math.pi/180*270])
    samples = tf.multinomial(tf.log([[10., 10.,10., 10.]]), 1) # note log-prob
    degree = elems[tf.cast(samples[0][0], tf.int32)]
    image = tf.contrib.image.rotate(image,degree)
    
    return image 
    
def distort_color(image, color_ordering=0, fast_mode=False, scope=None):
    """Distort the color of a Tensor image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
    Returns:
    3-D Tensor color-distorted image on range [0, 1]
    Raises:
    ValueError: if color_ordering not in [0, 3]
    """


    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=64. / 255.)
        image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
        image = tf.image.random_hue(image, max_delta=0.04)
        image = tf.image.random_contrast(image, lower=0.25, upper=1.75)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)

def random_zoom(image, cropped_height, cropped_width, output_shape = [224,224]):
    """Crops the given image using the provided offsets and sizes.

      Note that the method doesn't assume we know the input image size but it does
      assume we know the input image rank.

      Args:
        image: an image of shape [height, width, channels].
        offset_height: a scalar tensor indicating the height of cropped image.
        offset_width: a scalar tensor indicating the width of cropped image

      Returns:
        the cropped (and resized) image.

      Raises:
        InvalidArgumentError: if the rank is not 3 or if the image dimensions are
          less than the crop size.
      """

    # Randomly crop a [height, width] section of the image.
    cropped_image = tf.random_crop(image,
                                   [cropped_height, cropped_width, 3])
    cropped_image = tf.image.resize_images(cropped_image, output_shape)
    return cropped_image

def random_flip(image):
    # Randomly flip the image horizontally.
    flipped_image = tf.image.random_flip_left_right(image)
    return flipped_image

def shift(images, tx, ty, interpolation='NEAREST'):
    # got these parameters from solving the equations for pixel translations
    # on https://www.tensorflow.org/api_docs/python/tf/contrib/image/transform
    transforms = [1, 0, -tx, 0, 1, -ty, 0, 0]
    return tf.contrib.image.transform(images, transforms, interpolation)


