import tensorflow as tf
from tensorflow import keras

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-6

def rpn_loss_regr(num_anchors):
  def rpn_loss_regr_fixed_num(y_true, y_pred):
    x = y_true[:, :, :, 4*num_anchors:] - y_pred # coords
    x_abs = tf.abs(x)
    x_bool = tf.cast(tf.less_equal(x_abs, 1.0), tf.float32) # |x| < 1
    return lambda_rpn_regr * tf.reduce_sum(y_true[:, :, :, :4*num_anchors] * (x_bool * 0.5 * x * x + (1-x_bool) * (x_abs-0.5))) / tf.reduce_sum(epsilon+y_true[:, :, :, :4*num_anchors])
  return rpn_loss_regr_fixed_num

def rpn_loss_cls(num_anchors):
  def rpn_loss_cls_fixed_num(y_true, y_pred):
    # y_true[:, :, :, :num_anchors] decides whether should be a valid box (bg or fg)
    # y_true[:, :, :, num_anchors:] decides whether should be a frontground
    # mul = y_true[:, :, :, :num_anchors] * keras.losses.binary_crossentropy(y_true[:, :, :, num_anchors:], y_pred)
    sum = tf.reduce_sum(y_true[:, :, :, :num_anchors] * tf.losses.sigmoid_cross_entropy(y_true[:, :, :, num_anchors:], y_pred))
    return lambda_rpn_class * sum / tf.reduce_sum(epsilon+y_true[:, :, :, :num_anchors])
  return rpn_loss_cls_fixed_num

def class_loss_regr(num_classes):
  def class_loss_regr_fixed_num(y_true, y_pred):
    x = y_true[:, :, 4*num_classes:] - y_pred # coords
    x_abs = tf.abs(x)
    x_bool = tf.cast(tf.less_equal(x_abs, 1.0), tf.float32) # |x| < 1
    return lambda_cls_regr * tf.reduce_sum(y_true[:, :, :4*num_classes] * (x_bool * 0.5 * x * x + (1-x_bool) * (x_abs-0.5))) / tf.reduce_sum(epsilon+y_true[:, :, :4*num_classes])
  return class_loss_regr_fixed_num

def class_loss_cls(y_true, y_pred):
  print(y_pred.shape, y_true.shape)
  # return lambda_cls_class * tf.reduce_mean(keras.losses.categorical_crossentropy(y_true[:, :], y_pred[:, :]))
  return lambda_cls_class * tf.reduce_mean(tf.losses.softmax_cross_entropy(y_true[0], y_pred[0]))