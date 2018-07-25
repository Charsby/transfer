# Copyright 2016 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Definition of 300 DSOD network.

This model was initially introduced in:
SSD: Single Shot MultiBox Detector
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
Cheng-Yang Fu, Alexander C. Berg
https://arxiv.org/abs/1512.02325

Two variants of the model are defined: the 300x300 and 512x512 models, the
latter obtaining a slightly better accuracy on Pascal VOC.

Usage:
    with slim.arg_scope(ssd_vgg.ssd_vgg()):
        outputs, end_points = ssd_vgg.ssd_vgg(inputs)

This network port of the original Caffe model. The padding in TF and Caffe
is slightly different, and can lead to severe accuracy drop if not taken care
in a correct way!

In Caffe, the output size of convolution and pooling layers are computing as
following: h_o = (h_i + 2 * pad_h - kernel_h) / stride_h + 1

Nevertheless, there is a subtle difference between both for stride > 1. In
the case of convolution:
    top_size = floor((bottom_size + 2*pad - kernel_size) / stride) + 1
whereas for pooling:
    top_size = ceil((bottom_size + 2*pad - kernel_size) / stride) + 1
Hence implicitely allowing some additional padding even if pad = 0. This
behaviour explains why pooling with stride and kernel of size 2 are behaving
the same way in TensorFlow and Caffe.

Nevertheless, this is not the case anymore for other kernel sizes, hence
motivating the use of special padding layer for controlling these side-effects.

@@ssd_vgg_300
"""
import math
from collections import namedtuple

import numpy as np
import tensorflow as tf

import tf_extended as tfe
import custom_layers
import ssd_common
import dsod_utils

slim = tf.contrib.slim




# =========================================================================== #
# SSD class definition.
# =========================================================================== #
SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])


class DSODNet(object):
    """Implementation of the SSD VGG-based 300 network.

    The default features layers with 300x300 image input are:
      conv4 ==> 38 x 38
      conv7 ==> 19 x 19
      conv8 ==> 10 x 10
      conv9 ==> 5 x 5
      conv10 ==> 3 x 3
      conv11 ==> 1 x 1
    The default image size used to train this network is 300x300.
    """
    default_params = SSDParams(
        img_shape=(300, 300),
        num_classes=21,
        no_annotation_label=21,
        feat_layers=['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth'],
        feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
        anchor_size_bounds=[0.15, 0.90],
        # anchor_size_bounds=[0.20, 0.90],
        anchor_sizes=[(21., 45.),
                      (45., 99.),
                      (99., 153.),
                      (153., 207.),
                      (207., 261.),
                      (261., 315.)],
        # anchor_sizes=[(30., 60.),
        #               (60., 111.),
        #               (111., 162.),
        #               (162., 213.),
        #               (213., 264.),
        #               (264., 315.)],
        anchor_ratios=[[2, .5],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5],
                       [2, .5]],
        anchor_steps=[8, 16, 32, 64, 100, 300],
        anchor_offset=0.5,
        normalizations=[20, -1, -1, -1, -1, -1],
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
        )

    def __init__(self, params=None):
        """Init the SSD net with some parameters. Use the default ones
        if none provided.
        """
        if isinstance(params, SSDParams):
            self.params = params
        else:
            self.params = DSODNet.default_params

    # ======================================================================= #
    def net(self, inputs,
            is_training=True,
            update_feat_shapes=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='dsod'):
        """SSD network definition.
        """
        r = dsod_net(inputs,
                    num_classes=self.params.num_classes,
                    feat_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    normalizations=self.params.normalizations,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    scope=scope)
        # Update feature shapes (try at least!)
        if update_feat_shapes:
            shapes = ssd_feat_shapes_from_net(r[0], self.params.feat_shapes)
            self.params = self.params._replace(feat_shapes=shapes)
        return r

    def arg_scope(self, weight_decay=0.0005, data_format='NHWC'):
        """Network arg_scope.
        """
        return dsod_net_arg_scope(weight_decay)

    def arg_scope_caffe(self, caffe_scope):
        """Caffe arg_scope used for weights importing.
        """
        return dsod_net_arg_scope(caffe_scope)

    # ======================================================================= #
    def update_feature_shapes(self, predictions):
        """Update feature shapes from predictions collection (Tensor or Numpy
        array).
        """
        shapes = ssd_feat_shapes_from_net(predictions, self.params.feat_shapes)
        self.params = self.params._replace(feat_shapes=shapes)

    def anchors(self, img_shape, dtype=np.float32):
        """Compute the default anchor boxes, given an image shape.
        """
        return ssd_anchors_all_layers(img_shape,
                                      self.params.feat_shapes,
                                      self.params.anchor_sizes,
                                      self.params.anchor_ratios,
                                      self.params.anchor_steps,
                                      self.params.anchor_offset,
                                      dtype)

    def bboxes_encode(self, labels, bboxes, anchors,
                      scope=None):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_encode(
            labels, bboxes, anchors,
            self.params.num_classes,
            self.params.no_annotation_label,
            ignore_threshold=0.5,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def bboxes_decode(self, feat_localizations, anchors,
                      scope='ssd_bboxes_decode'):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_decode(
            feat_localizations, anchors,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def detected_bboxes(self, predictions, localisations,
                        select_threshold=None, nms_threshold=0.5,
                        clipping_bbox=None, top_k=400, keep_top_k=200):
        """Get the detected bounding boxes from the SSD network output.
        """
        # Select top_k bboxes from predictions, and clip
        rscores, rbboxes = \
            ssd_common.tf_ssd_bboxes_select(predictions, localisations,
                                            select_threshold=select_threshold,
                                            num_classes=self.params.num_classes)
        rscores, rbboxes = \
            tfe.bboxes_sort(rscores, rbboxes, top_k=top_k)
        # Apply NMS algorithm.
        rscores, rbboxes = \
            tfe.bboxes_nms_batch(rscores, rbboxes,
                                 nms_threshold=nms_threshold,
                                 keep_top_k=keep_top_k)
        if clipping_bbox is not None:
            rbboxes = tfe.bboxes_clip(clipping_bbox, rbboxes)
        return rscores, rbboxes

    def losses(self, logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope='ssd_losses'):
        """Define the SSD network losses.
        """
        return ssd_losses(logits, localisations,
                          gclasses, glocalisations, gscores,
                          match_threshold=match_threshold,
                          negative_ratio=negative_ratio,
                          alpha=alpha,
                          label_smoothing=label_smoothing,
                          scope=scope)




# =========================================================================== #
# SSD tools...
# =========================================================================== #
def ssd_size_bounds_to_values(size_bounds,
                              n_feat_layers,
                              img_shape=(300, 300)):
    """Compute the reference sizes of the anchor boxes from relative bounds.
    The absolute values are measured in pixels, based on the network
    default size (300 pixels).

    This function follows the computation performed in the original
    implementation of SSD in Caffe.

    Return:
      list of list containing the absolute sizes at each scale. For each scale,
      the ratios only apply to the first value.
    """
    assert img_shape[0] == img_shape[1]

    img_size = img_shape[0]
    min_ratio = int(size_bounds[0] * 100)
    max_ratio = int(size_bounds[1] * 100)
    step = int(math.floor((max_ratio - min_ratio) / (n_feat_layers - 2)))
    # Start with the following smallest sizes.
    sizes = [[img_size * size_bounds[0] / 2, img_size * size_bounds[0]]]
    for ratio in range(min_ratio, max_ratio + 1, step):
        sizes.append((img_size * ratio / 100.,
                      img_size * (ratio + step) / 100.))
    return sizes


def ssd_feat_shapes_from_net(predictions, default_shapes=None):
    """Try to obtain the feature shapes from the prediction layers. The latter
    can be either a Tensor or Numpy ndarray.

    Return:
      list of feature shapes. Default values if predictions shape not fully
      determined.
    """
    feat_shapes = []
    for l in predictions:
        # Get the shape, from either a np array or a tensor.
        if isinstance(l, np.ndarray):
            shape = l.shape
        else:
            shape = l.get_shape().as_list()
        shape = shape[1:4]
        # Problem: undetermined shape...
        if None in shape:
            return default_shapes
        else:
            feat_shapes.append(shape)
    return feat_shapes


def ssd_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
    """Computer SSD default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    # Compute the position grid: simple way.
    # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    # y = (y.astype(dtype) + offset) / feat_shape[0]
    # x = (x.astype(dtype) + offset) / feat_shape[1]
    # Weird SSD-Caffe computation using steps values...
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes) + len(ratios)
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)
    # Add first anchor boxes with ratio=1.
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):
        h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
    return y, x, h, w


def ssd_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = ssd_anchor_one_layer(img_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors


# =========================================================================== #
# Functional definition of VGG-based SSD 300.
# =========================================================================== #
def tensor_shape(x, rank=3):
    """Returns the dimensions of a tensor.
    Args:
      image: A N-D Tensor of shape.
    Returns:
      A list of dimensions. Dimensions that are statically known are python
        integers,otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]

    
def se_layer(input_tensor, output_channel, ratio = 4):
    with tf.name_scope("se_block") :
        # Squeeze and excitation in channel level
        squeeze = tf.reduce_mean(input_tensor, [1,2])
        excitation = tf.layers.dense(inputs=squeeze, use_bias=False, units=output_channel / ratio)
        excitation = tf.nn.relu(excitation)
        excitation = tf.layers.dense(inputs=excitation, use_bias=False, units=output_channel)
        excitation = tf.nn.sigmoid(excitation)
        excitation_loc = tf.reshape(excitation, [-1,1,1,output_channel])
        # global level
        excitation_glob = tf.layers.dense(inputs=excitation, use_bias=False, units=1)
        excitation_glob = tf.nn.sigmoid(excitation_glob)
        excitation_glob = tf.reshape(excitation_glob, [-1,1,1,1])
        
        # excitation in two levels
        scale_loc = input_tensor * excitation_loc
        scale_glob = scale_loc*excitation_glob
        #print excitation_loc, input_tensor,excitation_glob
        
        # residual block
        output = scale_glob+input_tensor
        
        return output


def ssd_multibox_layer(inputs,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False):
    """Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs
    if normalization > 0:
        net = custom_layers.l2_normalization(net, scaling=True)
    # Number of anchors.
    num_anchors = len(sizes) + len(ratios)

    # Location.
    num_loc_pred = num_anchors * 4
    loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None,
                           scope='conv_loc')
    loc_pred = custom_layers.channel_to_last(loc_pred)
    loc_pred = tf.reshape(loc_pred,
                          tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 4])
    # Class prediction.
    num_cls_pred = num_anchors * num_classes
    cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None,
                           scope='conv_cls')
    cls_pred = custom_layers.channel_to_last(cls_pred)
    cls_pred = tf.reshape(cls_pred,
                          tensor_shape(cls_pred, 4)[:-1]+[num_anchors, num_classes])
    return cls_pred, loc_pred


slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
xavier = tf.contrib.layers.xavier_initializer()

def dsod_net(inputs,
            num_classes,
            feat_layers,
            anchor_sizes,
            anchor_ratios,
            normalizations,
            dropout_keep_prob = 0.5,
            first_output = 128,
            growth_rate = 48,
            nchannels = 12,
            is_training = True,
            min_depth=16,
            depth_multiplier=1.0,
            use_separable_conv=True,
            data_format="NHWC",
            prediction_fn=slim.softmax,
            reuse=False,
            scope=None):
    """Dsod v1
    
    An tensorflow implementation of Dsod model described in https://github.com/szq0214/DSOD
    
    Args:
        inputs: a tensor of shape [batch_size, height, width, channels]
        min_depth: Minimum depth value (number of channels) for all convolution ops.
          Enforced when depth_multiplier < 1, and not an active constraint when 
          depth_multiplier >= 1.
        depth_multiplier: Float multiplier for the depth (number of channels)
          for all convolution ops. The value must be greater than zero. Typical
          usage will be to set this value in (0, 1) to reduce the number of
          parameters or computation cost of the model.
        data_format: Data format of the activations ('NHWC' or 'NCHW').
        scope: Optional variable_scope.
    """
    # end_points will collect relevant activations for external use, for example
    # summaries or losses.
    end_points = {}
    
    # Used to find thinned depths for each layer.
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    depth = lambda d: max(int(d * depth_multiplier), min_depth)

    if data_format != 'NHWC' and data_format != 'NCHW':
        raise ValueError('data_format must be either NHWC or NCHW.')
    if data_format == 'NCHW' and use_separable_conv:
        raise ValueError(
            'separable convolution only supports NHWC layout. NCHW data format can'
            ' only be used when use_separable_conv is False.'
        )

    concat_dim = 3 if data_format == 'NHWC' else 1

    # TODO: add name
    def bn_relu_conv(bottom, ks, nout, stride, pad, dropout, is_training,  scope="bn_relu_conv"):
        batch_norm = slim.batch_norm(bottom, decay=0.9997,
                                     epsilon=0.001,
                                     is_training= is_training,
                                     updates_collections=tf.GraphKeys.UPDATE_OPS,scope=scope)
        relu = tf.nn.relu(batch_norm)
        conv = slim.conv2d(
            relu,
            depth(nout),
            [ks, ks],
            padding=pad,
            stride=stride,
            weights_initializer=xavier,
            scope=scope)
        if dropout > 0:
            conv = tf.layers.dropout(conv, rate = dropout,training = is_training)
        return conv

    # TODO: add name
    def add_layer(bottom, num_filter, dropout,is_training, scope="add_layer"):
        conv = bn_relu_conv(bottom, ks=3, nout=num_filter, stride=1, pad="SAME", dropout=dropout, scope=scope, is_training = is_training)
        concate = tf.concat([bottom, conv], axis=concat_dim)
        return concate

    # TODO: add name, change num_filter
    def add_bl_layer(bottom, num_filter, dropout, width, is_training,scope="add_bl_layer"):
        conv = bn_relu_conv(bottom, ks=1, nout=int(width*num_filter), stride=1, pad="SAME", dropout=dropout, scope=scope+"_1", is_training = is_training)
        conv = bn_relu_conv(conv, ks=3, nout=num_filter, stride=1, pad="SAME", dropout=dropout, scope=scope+"_2", is_training = is_training)
        concate = tf.concat([bottom, conv], axis=concat_dim)
        return concate

    def add_bl_layer2(bottom, num_filter, dropout, width, is_training, scope="add_bl_layer2"):
        conv = bn_relu_conv(bottom, ks=1, nout=int(width * num_filter), stride=1, pad="SAME", dropout=dropout, scope=scope+"_1", is_training = is_training)
        conv = bn_relu_conv(conv, ks=3, nout=num_filter, stride=2, pad="SAME", dropout=dropout, scope=scope+"_2", is_training = is_training)
        conv2 = slim.max_pool2d(bottom, [2, 2], stride=2)
        conv2 = bn_relu_conv(conv2, ks=1, nout=num_filter, stride=1, pad="SAME", dropout=dropout, scope=scope+"_3", is_training = is_training)
        concate = tf.concat([conv2, conv], axis=concat_dim)
        return concate
    
    def add_bl_layer3(bottom, num_filter, dropout, width, is_training, scope="add_bl_layer3"):
        conv = bn_relu_conv(bottom, ks=1, nout=int(width * num_filter), stride=1, pad="SAME", dropout=dropout, scope=scope+"_1", is_training = is_training)
        conv = bn_relu_conv(conv, ks=3, nout=num_filter, stride=2, pad="VALID", dropout=dropout, scope=scope+"_2", is_training = is_training)
        conv2 = slim.max_pool2d(bottom, [2, 2], stride=2,padding = 'VALID')
        conv2 = bn_relu_conv(conv2, ks=1, nout=num_filter, stride=1, pad="SAME", dropout=dropout, scope=scope+"_3", is_training = is_training)
        concate = tf.concat([conv2, conv], axis=concat_dim)
        return concate

    def transition(bottom, num_filter, dropout, is_training,scope="transition"):
        conv = bn_relu_conv(bottom, ks=1, nout=num_filter, stride=1, pad="SAME", dropout=dropout, scope=scope, is_training = is_training)
        pooling = slim.max_pool2d(conv, [2, 2], stride=2)
        return pooling

    def transition3x3(bottom, num_filter, dropout,is_training, scope="transition3x3"):
        conv = bn_relu_conv(bottom, ks=3, nout=num_filter, stride=2, pad="VALID", dropout=dropout, scope=scope, is_training = is_training)
        # pooling = slim.max_pool2d(conv, [2, 2], stride=2)
        return conv

    def transition_w_o_pooling(bottom, num_filter, dropout,is_training, scope="transition_w_o_pooling"):
        conv = bn_relu_conv(bottom, ks=1, nout=num_filter, stride=1, pad="SAME", dropout=dropout, scope=scope, is_training = is_training)
        return conv

    # Stem
    with tf.variable_scope(scope, 'dsodV1', [inputs]) as glob:
        if reuse:
            glob.reuse_variables()
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding="SAME", data_format=data_format):

            # 150x150
            end_point = "conv1"
            if use_separable_conv:
                depthwise_multiplier = min(int(depth(64) / 3), 8)
                net = slim.separable_conv2d(inputs, depth(64), [3, 3],
                                            depth_multiplier=depthwise_multiplier,
                                            stride=2, padding="SAME",
                                            weights_initializer=xavier)
            else:
                net = slim.conv2d(inputs, depth(64), [3, 3],
                                  stride=2,
                                  weights_initializer=xavier,
                                  scope=end_point)
            end_points[end_point] = net # output: 150x150

            end_point = "conv2"
            net = slim.conv2d(net, depth(64), [3, 3],
                              weights_initializer=xavier,
                              scope=end_point)
            end_points[end_point] = net

            end_point = "conv3"
            net = slim.conv2d(net, depth(128), [3, 3],
                              weights_initializer=xavier,
                              scope=end_point)
            end_points[end_point] = net

            # 75x75
            end_point = "pooling1"
            net = slim.max_pool2d(net, [2, 2], stride=2)
            end_points[end_point] = net


            times = 1
            for i in range(6):
                net = add_bl_layer(net, growth_rate, dropout_keep_prob, 4, scope = "first/a_"+str(i), is_training = is_training)
                nchannels += growth_rate
            nchannels = int(nchannels / times)
            net = transition(net, nchannels, dropout_keep_prob,is_training = is_training) # pooling2: 37x37 (for caffe: 38x38)
            for i in range(8):
                net = add_bl_layer(net, growth_rate, dropout_keep_prob, 4, scope ="first/b_"+str(i), is_training = is_training)
                nchannels += growth_rate
            nchannels = int(nchannels / times)

            end_point = "First"
            net = transition_w_o_pooling(net, nchannels, dropout_keep_prob, scope ="first/transition_w_o_pooling", is_training = is_training) # 37x37
            end_points[end_point] = net

            end_point = "Second"
            net1 = slim.max_pool2d(net, [2, 2], stride=2)
            for i in range(8):
                net1 = add_bl_layer(net1, growth_rate, dropout_keep_prob, 4, scope ="second/a_"+str(i), is_training = is_training)
                nchannels += growth_rate
            nchannels = int(nchannels / times)
            net1 = transition_w_o_pooling(net1, nchannels, dropout_keep_prob, is_training = is_training)
            for i in range(8):
                net1 = add_bl_layer(net1, growth_rate, dropout_keep_prob, 4, scope ="second/b_"+str(i), is_training = is_training)
                nchannels += growth_rate
            # nchannels = int(nchannels / times)
            net1 = transition_w_o_pooling(net1, 256, dropout_keep_prob, scope = "second/transition_w_o_pooling", is_training = is_training)

            f_first = slim.max_pool2d(net, [2, 2], stride=2)
            f_first = bn_relu_conv(f_first, ks=1, nout=256, stride=1, pad="SAME", dropout=dropout_keep_prob, scope="sceond/f_first", is_training = is_training)
            net1 = tf.concat([net1, f_first], axis=concat_dim) # 18x18
            end_points[end_point] = net1

            end_point = "Third"
            net2 = add_bl_layer2(net1, 256, dropout_keep_prob, 1, scope ="third/add_bl_layer2", is_training = is_training) # 9x9
            end_points[end_point] = net2

            end_point = "Fourth"
            net3 = add_bl_layer2(net2, 128, dropout_keep_prob, 1, scope ="fourth/add_bl_layer2", is_training = is_training) # 4x4
            end_points[end_point] = net3

            end_point = "Fifth"
            net4 = add_bl_layer2(net3, 128, dropout_keep_prob, 1,scope = "fifth/add_bl_layer2", is_training = is_training) # 2x2
            end_points[end_point] = net4

            end_point = "Sixth"
            net5 = add_bl_layer3(net4, 128, dropout_keep_prob, 1, scope ="sixth/add_bl_layer2", is_training = is_training)  # 1x1
            end_points[end_point] = net5
            
    
            predictions = []
            logits = []
            localisations = []
            output_channels = [684,512,512,256,256,256]
            for i, layer in enumerate(feat_layers):
                with tf.variable_scope(layer + '_box'):
                       
                    output_layer = end_points[layer]
                    output_layer = se_layer(output_layer, output_channels[i], ratio = 4)
                    
                    p, l = ssd_multibox_layer(output_layer,
                                              num_classes,
                                              anchor_sizes[i],
                                              anchor_ratios[i],
                                              normalizations[i])
                predictions.append(prediction_fn(p))
                logits.append(p)
                localisations.append(l)

            return predictions, localisations, logits, end_points
dsod_net.default_image_size = 300
dsod_net_arg_scope = dsod_utils.dsod_arg_scope


# =========================================================================== #
# SSD loss function.
# =========================================================================== #
def ssd_losses(logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               device='/cpu:0',
               scope=None):
    with tf.name_scope(scope, 'ssd_losses'):
        lshape = tfe.get_shape(logits[0], 5)
        num_classes = lshape[-1]
        batch_size = lshape[0]

        # Flatten out all vectors!
        flogits = []
        fgclasses = []
        fgscores = []
        flocalisations = []
        fglocalisations = []
        for i in range(len(logits)):
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))
            fgclasses.append(tf.reshape(gclasses[i], [-1]))
            fgscores.append(tf.reshape(gscores[i], [-1]))
            flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
            fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))
        # And concat the crap!
        logits = tf.concat(flogits, axis=0)
        gclasses = tf.concat(fgclasses, axis=0)
        gscores = tf.concat(fgscores, axis=0)
        localisations = tf.concat(flocalisations, axis=0)
        glocalisations = tf.concat(fglocalisations, axis=0)
        dtype = logits.dtype
        # Compute positive matching mask...
        pmask = gscores > match_threshold
        fpmask = tf.cast(pmask, dtype)
        n_positives = tf.reduce_sum(fpmask)

        # Hard negative mining...
        no_classes = tf.cast(pmask, tf.int32)
        predictions = slim.softmax(logits)
        nmask = tf.logical_and(tf.logical_not(pmask),
                               gscores > -0.5)
        fnmask = tf.cast(nmask, dtype)
        nvalues = tf.where(nmask,
                           predictions[:, 0],
                           1. - fnmask)
        nvalues_flat = tf.reshape(nvalues, [-1])
        # Number of negative entries to select.
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
        n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + batch_size
        n_neg = tf.minimum(n_neg, max_neg_entries)

        val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
        max_hard_pred = -val[-1]
        # Final negative mask.
        nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
        fnmask = tf.cast(nmask, dtype)
        
        # Add cross-entropy loss.
        with tf.name_scope('cross_entropy_pos'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=gclasses)
            loss_clss_pos = tf.div(tf.reduce_sum(loss * fpmask), batch_size, name='value')
            tf.losses.add_loss(loss_clss_pos)

        with tf.name_scope('cross_entropy_neg'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=no_classes)
            loss_class_neg = tf.div(tf.reduce_sum(loss * fnmask), batch_size, name='value')
            tf.losses.add_loss(loss_class_neg)

        # Add localization loss: smooth L1, L2, ...
        with tf.name_scope('localization'):
            # Weights Tensor: positive mask + random negative.
            weights = tf.expand_dims(alpha * fpmask, axis=-1)
            loss = custom_layers.abs_smooth(localisations - glocalisations)
            loss_loc = tf.div(tf.reduce_sum(loss * weights), batch_size, name='value')
            tf.losses.add_loss(loss_loc)
        tot_num = tf.cast(n_neg,tf.float32)+n_positives+tf.constant(1e-8,dtype=tf.float32)
        loss_clss = loss_clss_pos+loss_class_neg
        tot_loss =loss_clss +6*loss_loc
        ave_loss = tf.div(tot_loss,tot_num)*tf.cast(batch_size,tf.float32)
        return [loss_clss,loss_loc,tot_num,ave_loss]


def ssd_losses_old(logits, localisations,
                   gclasses, glocalisations, gscores,
                   match_threshold=0.5,
                   negative_ratio=3.,
                   alpha=1.,
                   label_smoothing=0.,
                   device='/cpu:0',
                   scope=None):
    """Loss functions for training the SSD 300 VGG network.

    This function defines the different loss components of the SSD, and
    adds them to the TF loss collection.

    Arguments:
      logits: (list of) predictions logits Tensors;
      localisations: (list of) localisations Tensors;
      gclasses: (list of) groundtruth labels Tensors;
      glocalisations: (list of) groundtruth localisations Tensors;
      gscores: (list of) groundtruth score Tensors;
    """
    with tf.device(device):
        with tf.name_scope(scope, 'ssd_losses'):
            l_cross_pos = []
            l_cross_neg = []
            l_loc = []
            for i in range(len(logits)):
                dtype = logits[i].dtype
                with tf.name_scope('block_%i' % i):
                    # Sizing weight...
                    wsize = tfe.get_shape(logits[i], rank=5)
                    wsize = wsize[1] * wsize[2] * wsize[3]

                    # Positive mask.
                    pmask = gscores[i] > match_threshold
                    fpmask = tf.cast(pmask, dtype)
                    n_positives = tf.reduce_sum(fpmask)

                    # Select some random negative entries.
                    # n_entries = np.prod(gclasses[i].get_shape().as_list())
                    # r_positive = n_positives / n_entries
                    # r_negative = negative_ratio * n_positives / (n_entries - n_positives)

                    # Negative mask.
                    no_classes = tf.cast(pmask, tf.int32)
                    predictions = slim.softmax(logits[i])
                    nmask = tf.logical_and(tf.logical_not(pmask),
                                           gscores[i] > -0.5)
                    fnmask = tf.cast(nmask, dtype)
                    nvalues = tf.where(nmask,
                                       predictions[:, :, :, :, 0],
                                       1. - fnmask)
                    nvalues_flat = tf.reshape(nvalues, [-1])
                    # Number of negative entries to select.
                    n_neg = tf.cast(negative_ratio * n_positives, tf.int32)
                    n_neg = tf.maximum(n_neg, tf.size(nvalues_flat) // 8)
                    n_neg = tf.maximum(n_neg, tf.shape(nvalues)[0] * 4)
                    max_neg_entries = 1 + tf.cast(tf.reduce_sum(fnmask), tf.int32)
                    n_neg = tf.minimum(n_neg, max_neg_entries)

                    val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
                    max_hard_pred = -val[-1]
                    # Final negative mask.
                    nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
                    fnmask = tf.cast(nmask, dtype)

                    # Add cross-entropy loss.
                    with tf.name_scope('cross_entropy_pos'):
                        fpmask = wsize * fpmask
                        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
                                                                              labels=gclasses[i])
                        loss = tf.losses.compute_weighted_loss(loss, fpmask)
                        l_cross_pos.append(loss)

                    with tf.name_scope('cross_entropy_neg'):
                        fnmask = wsize * fnmask
                        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
                                                                              labels=no_classes)
                        loss = tf.losses.compute_weighted_loss(loss, fnmask)
                        l_cross_neg.append(loss)

                    # Add localization loss: smooth L1, L2, ...
                    with tf.name_scope('localization'):
                        # Weights Tensor: positive mask + random negative.
                        weights = tf.expand_dims(alpha * fpmask, axis=-1)
                        loss = custom_layers.abs_smooth(localisations[i] - glocalisations[i])
                        loss = tf.losses.compute_weighted_loss(loss, weights)
                        l_loc.append(loss)

            # Additional total losses...
            with tf.name_scope('total'):
                total_cross_pos = tf.add_n(l_cross_pos, 'cross_entropy_pos')
                total_cross_neg = tf.add_n(l_cross_neg, 'cross_entropy_neg')
                total_cross = tf.add(total_cross_pos, total_cross_neg, 'cross_entropy')
                total_loc = tf.add_n(l_loc, 'localization')

                # Add to EXTRA LOSSES TF.collection
                tf.add_to_collection('EXTRA_LOSSES', total_cross_pos)
                tf.add_to_collection('EXTRA_LOSSES', total_cross_neg)
                tf.add_to_collection('EXTRA_LOSSES', total_cross)
                tf.add_to_collection('EXTRA_LOSSES', total_loc)
