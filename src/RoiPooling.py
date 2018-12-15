# roi pooling layer

import tensorflow as tf
from tensorflow import keras

class RoiPooling(keras.layers.Layer):
  '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        4D tensor with shape:
        `(1, rows, cols, channels)`
        X_roi:
        `(1, num_rois, 4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        5D tensor with shape:
        `(1, num_rois, pool_size, pool_size, channels)`
  '''
  def __init__(self, pool_size, num_rois, **kwargs):
    self.pool_size = pool_size
    self.num_rois = num_rois
    super(RoiPooling, self).__init__(**kwargs)
  
  def build(self, input_shape):
    self.n_channels = input_shape[0][3]
  
  def compute_output_shape(self, input_shape):
    return None, self.num_rois, self.pool_size, self.pool_size, self.n_channels

  # def call(self, x, mask=None):
  def call(self, x, mask=None):
    # assert(len(x) == 2)
    img = x[0]
    rois = x[1]
    outputs = []
    print(img.shape)
    print(rois.shape)

    # img = tf.transpose(img, (0, 3, 1, 2))
    # input_shape = tf.shape(img)

    for roi_idx in range(self.num_rois):
      x = rois[0, roi_idx, 0]
      y = rois[0, roi_idx, 1]
      w = rois[0, roi_idx, 2]
      h = rois[0, roi_idx, 3]
      print(x, y, w, h)

      # row_length = w / self.pool_size
      # col_length = h / self.pool_size

      # for jy in range(self.pool_size):
      #   for ix in range(self.pool_size):
      #     x1 = x + ix * row_length
      #     x2 = x1 + row_length
      #     y1 = y + jy * col_length
      #     y2 = y1 + col_length

      #     x1 = tf.cast(x1, tf.int32)
      #     x2 = tf.cast(x2, tf.int32)
      #     y1 = tf.cast(y1, tf.int32)
      #     y2 = tf.cast(y2, tf.int32)

      #     x2 = x1 + tf.maximum(1, x2-x1)
      #     y2 = y1 + tf.maximum(1, y2-y1)

      #     new_shape = [input_shape[0], input_shape[1], y2-y1, x2-x1]

      #     xm = tf.reshape(img[:, :, y1:y2, x2:x2], new_shape)
      #     pooled_val = tf.reduce_max(xm, axis=(2, 3))
      #     outputs.append(pooled_val)

      x = tf.cast(x, tf.int32)
      y = tf.cast(y, tf.int32)
      w = tf.cast(w, tf.int32)
      h = tf.cast(h, tf.int32)

      rs = tf.image.resize_images(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
      # rs = img[:, :self.pool_size, :self.pool_size, :]
      outputs.append(rs)
    
    final_output = tf.concat(outputs, axis=0)
    # final_output = keras.layers.Flatten()(img)
    # final_output = tf.slice(final_output, [0,0], [1,self.num_rois*self.pool_size*self.pool_size*self.n_channels])
    final_output = tf.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.n_channels))

    return final_output

  def get_config(self):
    config = {'pool_size': self.pool_size, 'num_rois': self.num_rois}
    base_config = super(RoiPooling, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


