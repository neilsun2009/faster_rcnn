# ResNet50

import tensorflow as tf
from tensorflow import keras
from RoiPooling import RoiPooling

WEIGHT_PATH = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

# imitate the procedure of resnet
# get the feature map w/h
def get_img_output_length(width, height):
  def get_output_length(input_length):
    # zero_pad
    input_length += 6
    # 4 strided operations
    # conv1, maxpool, conv3, conv4
    filter_sizes = [7, 3, 1, 1]
    stride = 2
    for filter_size in filter_sizes:
      input_length = (input_length - filter_size + stride) // stride
    return input_length
  return get_output_length(width), get_output_length(height)

# def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True):
#   filter1, filter2, filter3 = filters
#   bn_axis = 3
#   conv_name_base = 'res%s_branch' % (str(stage)+block)
#   bn_name_base = 'bn%s_branch' % (str(stage)+block)

#   x = keras.layers.Conv2D(filter1, (1, 1), name=conv_name_base+'2a',
#     padding='same', trainable=trainable)(input_tensor)
#   x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base+'2a')(x)
#   x = keras.layers.ReLU()(x)
  
#   x = keras.layers.Conv2D(filter2, kernel_size, name=conv_name_base+'2b',
#     padding='same', trainable=trainable)(x)
#   x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base+'2b')(x)
#   x = keras.layers.ReLU()(x)

#   x = keras.layers.Conv2D(filter3, (1, 1), name=conv_name_base+'2c',
#     padding='same', trainable=trainable)(x)
#   x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base+'2c')(x)

#   x = keras.layers.Add()([x, input_tensor])
#   x = keras.layers.ReLU()(x)
#   return x

def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True):
  filter1, filter2, filter3 = filters
  bn_axis = 3
  conv_name_base = 'res%s_branch' % (str(stage)+block)
  bn_name_base = 'bn%s_branch' % (str(stage)+block)

  x = tf.layers.Conv2D(filter1, (1, 1),
    padding='same', trainable=trainable, name=conv_name_base+'2a')(input_tensor)
  x = tf.layers.BatchNormalization(axis=bn_axis, name=bn_name_base+'2a')(x)
  x = tf.nn.relu(x)
  
  x = tf.layers.Conv2D(filter2, kernel_size,
    padding='same', trainable=trainable, name=conv_name_base+'2b')(x)
  x = tf.layers.BatchNormalization(axis=bn_axis, name=bn_name_base+'2b')(x)
  x = tf.nn.relu(x)

  x = tf.layers.Conv2D(filter3, (1, 1),
    padding='same', trainable=trainable, name=conv_name_base+'2c')(x)
  x = tf.layers.BatchNormalization(axis=bn_axis, name=bn_name_base+'2c')(x)

  x = tf.math.add(x, input_tensor)
  x = tf.nn.relu(x)
  return x

# def identity_block_td(input_tensor, kernel_size, filters, stage, block, trainable=True):
#   filter1, filter2, filter3 = filters
#   bn_axis = 3
#   conv_name_base = 'res%s_branch' % (str(stage)+block)
#   bn_name_base = 'bn%s_branch' % (str(stage)+block)

#   x = keras.layers.TimeDistributed(keras.layers.Conv2D(filter1, (1, 1),
#     padding='same', trainable=trainable), name=conv_name_base+'2a')(input_tensor)
#   x = keras.layers.TimeDistributed(keras.layers.BatchNormalization(axis=bn_axis), name=bn_name_base+'2a')(x)
#   x = keras.layers.ReLU()(x)
  
#   x = keras.layers.TimeDistributed(keras.layers.Conv2D(filter2, kernel_size,
#     padding='same', trainable=trainable), name=conv_name_base+'2b')(x)
#   x = keras.layers.TimeDistributed(keras.layers.BatchNormalization(axis=bn_axis), name=bn_name_base+'2b')(x)
#   x = keras.layers.ReLU()(x)

#   x = keras.layers.TimeDistributed(keras.layers.Conv2D(filter3, (1, 1),
#     padding='same', trainable=trainable), name=conv_name_base+'2c')(x)
#   x = keras.layers.TimeDistributed(keras.layers.BatchNormalization(axis=bn_axis), name=bn_name_base+'2c')(x)

#   x = keras.layers.Add()([x, input_tensor])
#   x = keras.layers.ReLU()(x)
#   return x

# def conv_block(input_tensor, kernel_size, filters, stage, block, input_shape=None, strides=(2, 2), trainable=True):
#   filter1, filter2, filter3 = filters
#   bn_axis = 3
#   conv_name_base = 'res%s_branch' % (str(stage)+block)
#   bn_name_base = 'bn%s_branch' % (str(stage)+block)

#   if input_shape is None:
#     x = keras.layers.Conv2D(filter1, (1, 1), strides=strides, name=conv_name_base+'2a',
#       padding='same', trainable=trainable)(input_tensor)
#   else:
#     x = keras.layers.Conv2D(filter1, (1, 1), strides=strides, name=conv_name_base+'2a',
#       padding='same', input_shape=input_shape, trainable=trainable)(input_tensor)
#   x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base+'2a')(x)
#   x = keras.layers.ReLU()(x)
  
#   x = keras.layers.Conv2D(filter2, kernel_size, name=conv_name_base+'2b',
#     padding='same', trainable=trainable)(x)
#   x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base+'2b')(x)
#   x = keras.layers.ReLU()(x)

#   x = keras.layers.Conv2D(filter3, (1, 1), name=conv_name_base+'2c',
#     padding='same', trainable=trainable)(x)
#   x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base+'2c')(x)

#   shortcut = keras.layers.Conv2D(filter3, (1, 1), strides=strides, name=conv_name_base + '1',
#     padding='same', trainable=trainable)(input_tensor)
#   shortcut = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base+'1')(shortcut)

#   x = keras.layers.Add()([x, shortcut])
#   x = keras.layers.ReLU()(x)
#   return x

def conv_block(input_tensor, kernel_size, filters, stage, block, input_shape=None, strides=(2, 2), trainable=True):
  filter1, filter2, filter3 = filters
  bn_axis = 3
  conv_name_base = 'res%s_branch' % (str(stage)+block)
  bn_name_base = 'bn%s_branch' % (str(stage)+block)

  if input_shape is None:
    x = tf.layers.Conv2D(filter1, (1, 1), strides=strides,
      padding='same', trainable=trainable, name=conv_name_base+'2a')(input_tensor)
  else:
    x = tf.layers.Conv2D(filter1, (1, 1), strides=strides,
      padding='same', trainable=trainable, name=conv_name_base+'2a', input_shape=input_shape)(input_tensor)
  x = tf.layers.BatchNormalization(axis=bn_axis, name=bn_name_base+'2a')(x)
  x = tf.nn.relu(x)
  
  x = tf.layers.Conv2D(filter2, kernel_size,
    padding='same', trainable=trainable, name=conv_name_base+'2b')(x)
  x = tf.layers.BatchNormalization(axis=bn_axis, name=bn_name_base+'2b')(x)
  x = tf.nn.relu(x)

  x = tf.layers.Conv2D(filter3, (1, 1),
    padding='same', trainable=trainable, name=conv_name_base+'2c')(x)
  x = tf.layers.BatchNormalization(axis=bn_axis, name=bn_name_base+'2c')(x)

  shortcut = tf.layers.Conv2D(filter3, (1, 1), strides=strides,
    padding='same', trainable=trainable, name=conv_name_base + '1')(input_tensor)
  shortcut = tf.layers.BatchNormalization(axis=bn_axis, name=bn_name_base+'1')(shortcut)

  x = tf.math.add(x, shortcut)
  x = tf.nn.relu(x)
  return x

# def conv_block_td(input_tensor, kernel_size, filters, stage, block, input_shape, strides=(2, 2), trainable=True):
#   filter1, filter2, filter3 = filters
#   bn_axis = 3
#   conv_name_base = 'res%s_branch' % (str(stage)+block)
#   bn_name_base = 'bn%s_branch' % (str(stage)+block)

#   x = keras.layers.TimeDistributed(keras.layers.Conv2D(filter1, (1, 1), strides=strides,
#     padding='same', trainable=trainable), name=conv_name_base+'2a', input_shape=input_shape)(input_tensor)
#   x = keras.layers.TimeDistributed(keras.layers.BatchNormalization(axis=bn_axis), name=bn_name_base+'2a')(x)
#   x = keras.layers.ReLU()(x)
  
#   x = keras.layers.TimeDistributed(keras.layers.Conv2D(filter2, kernel_size,
#     padding='same', trainable=trainable), name=conv_name_base+'2b')(x)
#   x = keras.layers.TimeDistributed(keras.layers.BatchNormalization(axis=bn_axis), name=bn_name_base+'2b')(x)
#   x = keras.layers.ReLU()(x)

#   x = keras.layers.TimeDistributed(keras.layers.Conv2D(filter3, (1, 1),
#     padding='same', trainable=trainable), name=conv_name_base+'2c')(x)
#   x = keras.layers.TimeDistributed(keras.layers.BatchNormalization(axis=bn_axis), name=bn_name_base+'2cs')(x)

#   shortcut = keras.layers.TimeDistributed(keras.layers.Conv2D(filter3, (1, 1), strides=strides,
#     padding='same', trainable=trainable), name=conv_name_base + '1')(input_tensor)
#   shortcut = keras.layers.TimeDistributed(keras.layers.BatchNormalization(axis=bn_axis), name=bn_name_base+'1')(shortcut)

#   x = keras.layers.Add()([x, shortcut])
#   x = keras.layers.ReLU()(x)
#   return x

def nn_base(input_tensor, trainable=False):
  # if input_tensor is None:
  #   img_input = keras.layers.Input(shape=(None, None, 3))
  # else:
  #   img_input = keras.layers.Input(tensor=input_tensor, shape=(None, None, 3))
  
  bn_axis=3

  # conv1
  x = tf.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same',
    name='conv1', trainable=trainable)(input_tensor)
  x = tf.layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
  x = tf.nn.relu(x)
  x = tf.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

  # conv2
  x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), trainable=trainable)
  x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', trainable=trainable)
  x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', trainable=trainable)

  # conv3
  x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', trainable=trainable)
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', trainable=trainable)
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', trainable=trainable)
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', trainable=trainable)

  # conv4
  x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', trainable=trainable)
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', trainable=trainable)
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', trainable=trainable)
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', trainable=trainable)
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', trainable=trainable)
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', trainable=trainable)

  return x

# def classifier(roi_pool, input_shape, n_classes=21, trainable=False):
#   # roi pool: output from roi pooling
#   x = conv_block_td(roi_pool, 3, [512, 512, 2048], input_shape=input_shape, stage=5, block='a', trainable=trainable)
#   x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
#   x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='c', trainable=trainable)
  
#   out = keras.layers.TimeDistributed(keras.layers.AveragePooling2D((7, 7)), name='avg_pool')(x)
#   out = keras.layers.TimeDistributed(keras.layers.Flatten())(out)

#   out_class = keras.layers.TimeDistributed(keras.layers.Dense(n_classes, activation='softmax', kernel_initializer='zero'),
#     name='dense_class_'+str(n_classes))(out)
#   # no regerssion for bg class
#   out_regress = keras.layers.TimeDistributed(keras.layers.Dense(4*(n_classes-1), activation='linear', kernel_initializer='zero'),
#     name='dense_regress_'+str(n_classes))(out)
#   # out_class = tf.expand_dims(out_class, axis=0)
#   # out_regress = tf.expand_dims(out_regress, axis=0)
#   return [out_class, out_regress]

def roi(img, rois, pool_size, num_rois):
  n_channels = img.shape[-1]
  roi_pool = []
  for roi_idx in range(num_rois):
      x = rois[0, roi_idx, 0]
      y = rois[0, roi_idx, 1]
      w = rois[0, roi_idx, 2]
      h = rois[0, roi_idx, 3]

      x = tf.cast(x, tf.int32)
      y = tf.cast(y, tf.int32)
      w = tf.cast(w, tf.int32)
      h = tf.cast(h, tf.int32)

      rs = tf.image.resize_images(img[:, y:y+h, x:x+w, :], (pool_size, pool_size))
      # rs = img[:, :pool_size, :pool_size, :]
      roi_pool.append(rs)
  roi_pool = tf.concat(roi_pool, axis=0)
  # final_output = keras.layers.Flatten()(img)
  # final_output = tf.slice(final_output, [0,0], [1,self.num_rois*self.pool_size*self.pool_size*self.n_channels])
  roi_pool = tf.reshape(roi_pool, (num_rois, pool_size, pool_size, n_channels))
  return roi_pool

# def classifier(roi_pool, input_shape, n_classes=21, trainable=False):
def classifier(img, rois, num_rois, input_shape, n_classes=21, trainable=False):
  # roi pooling
  roi_pool = RoiPooling(14, num_rois)([img, rois])
  # roi_pool = roi(img, rois, 14, num_rois)
  # roi pool: output from roi pooling
  roi_pool = tf.squeeze(roi_pool, [0])
  x = conv_block(roi_pool, 3, [512, 512, 2048], input_shape=input_shape, stage=5, block='a', trainable=trainable)
  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', trainable=trainable)
  
  out = tf.layers.AveragePooling2D((7, 7), (7, 7), name='avg_pool')(x)
  out = tf.layers.Flatten()(out)

  out = tf.reshape(out, (1, num_rois, -1))

  # out_class = keras.layers.TimeDistributed(keras.layers.Dense(n_classes, activation='softmax', kernel_initializer='zero'),
  #   name='dense_class_'+str(n_classes))(out)
  # # no regerssion for bg class
  # out_regress = keras.layers.TimeDistributed(keras.layers.Dense(4*(n_classes-1), activation='linear', kernel_initializer='zero'),
  #   name='dense_regress_'+str(n_classes))(out)
  out_class = tf.layers.Dense(n_classes, kernel_initializer='zero',
    name='dense_class_'+str(n_classes))(out)
  # no regerssion for bg class
  out_regress = tf.layers.Dense(4*(n_classes-1), kernel_initializer='zero',
    name='dense_regress_'+str(n_classes))(out)
  # out_class = tf.expand_dims(out_class, axis=0)
  # out_regress = tf.expand_dims(out_regress, axis=0)
  return [out_class, out_regress]
