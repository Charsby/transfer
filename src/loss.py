import tensorflow as tf
from keras import backend as K
from tensorflow.python.ops import array_ops

class Loss(object):
    """Loss functions.
    # Arguments
        num_classes: Number of classes including background.
        alpha: Weight of L1-smooth loss.
        negatives_for_hard: Number of hard negative examples.
    """
    def __init__(self, num_classes = 2, alpha=1.0, negatives_for_hard = 2):
        print ('new1')
        self.num_classes = num_classes
        self.alpha = alpha
        self.negatives_for_hard = negatives_for_hard
         
    def _l1_smooth_loss(self, y_true, y_pred):
        """Compute L1-smooth loss.

        # Arguments
            y_true: Ground truth bounding boxes,
            y_pred: Predicted bounding boxes,
        # Returns
            l1_loss: L1-smooth loss

        # References
            https://arxiv.org/abs/1504.08083
        """
        abs_loss = tf.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
        return tf.reduce_sum(l1_loss, -1)

    def _softmax_loss(self, y_true, y_pred):
        """Compute softmax loss.

        # Arguments
            y_true: Ground truth targets,
            y_pred: Predicted logits,

        # Returns
            softmax_loss: Softmax loss.
        """
        y_true = tf.greater(y_true, 0)
        y_true = tf.to_int32(y_true)
        y_pred = tf.maximum(tf.minimum(y_pred, 1 - 1e-15), 1e-15)
        softmax_loss = -tf.reduce_sum(y_true * tf.log(y_pred),
                                      axis=-1)
        return softmax_loss

    def ohnm_loss(self, y_true, y_pred):
        """Compute the ohnm loss.
        # Arguments
            y_true: Ground truth targets,
            y_pred: Predicted logits,
        # Returns
            total loss: Total Loss
        """
              
        num_hard = self.negatives_for_hard
        batch_size = tf.shape(y_true)[0]
        num_boxes = tf.to_int32(tf.shape(y_true)[1] * tf.shape(y_true)[2] * tf.shape(y_true)[3])
        num_anchors = tf.to_int32(tf.shape(y_true)[4])
        
        y_pred = tf.reshape(y_pred, [-1, 5])
        y_true = tf.reshape(y_true, [-1, 5])
        
        y_prob = K.sigmoid(y_pred)
        
        # loss for all priors
        
        cls_loss = K.binary_crossentropy(tf.to_float(tf.greater(y_true[:, 0],0)), y_prob[:, 0]) / 2
        reg_loss = self._l1_smooth_loss(y_true[:, 1:], y_pred[:, 1:])
        
        # get positives loss
        num_pos = tf.reduce_sum(tf.to_int32(tf.greater(y_true[:, 0],0)), axis=-1)
        pos_reg_loss = tf.reduce_sum(reg_loss * tf.to_float(tf.greater(y_true[:, 0],0)), axis = -1)
        pos_cls_loss = tf.reduce_sum(cls_loss * tf.to_float(tf.greater(y_true[:, 0],0)), axis = -1)
        
        # get negatives loss, we penalize only confidence here
        num_neg = tf.reduce_sum(tf.to_int32(tf.less(y_true[:, 0],0)), axis=-1)
        # OHNM
        _, indices = tf.nn.top_k(y_prob[:, 0] * (tf.to_float(tf.less(y_true[:, 0],0))), k = num_hard)
        
        neg_output = K.mean(tf.gather(y_pred[:, 0], indices))
        num_neg = tf.minimum(num_hard, num_neg)
        
        neg_cls_loss = tf.reduce_sum(tf.gather(cls_loss, indices), axis = -1) 
        
        #pos_loss = K.switch(K.not_equal(num_pos, 0), (pos_cls_loss + pos_reg_loss) / tf.to_float(num_pos), K.constant(0) * (pos_cls_loss + pos_reg_loss))
        #neg_loss = (neg_cls_loss) / tf.to_float(num_neg)
        
        total_loss = (pos_cls_loss + pos_reg_loss + neg_cls_loss) / (tf.to_float(num_neg) + tf.to_float(num_pos))
        
        return total_loss
    
    def focal_loss(self, y_true, y_pred, weights=None, alpha=0.25, gamma=2):
        r"""Compute focal loss for predictions.
            Multi-labels Focal loss formula:
                FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                     ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
        Args:
         y_true: Ground truth targets,
         y_pred: Predicted logits,
         weights: A float tensor of shape [batch_size, num_anchors]
         alpha: A scalar tensor for focal loss alpha hyper-parameter
         gamma: A scalar tensor for focal loss gamma hyper-parameter
        Returns:
            loss: A (scalar) tensor representing the value of the loss function
        """
        sigmoid_p = tf.nn.sigmoid(y_pred)
        zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
        pos_p_sub = array_ops.where(target_tensor >= sigmoid_p, target_tensor - sigmoid_p, zeros)
        neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
        per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                              - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
        return tf.reduce_mean(per_entry_cross_ent)
        
        return total_loss