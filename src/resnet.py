# ResNet50

import tensorflow
from tensorflow import keras

WEIGHT_PATH = 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'

def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True):
  filter1, filter2, filter3 = filters
  bn_axis = 1
  conv_name_base = 'res%s_branch' % (str(stage)+block)
  bn_name_base = 'bn%s_branch' % (str(stage)+block)

  x = keras.layers.Conv2D(filter1, (1, 1), name=conv_name_base+'2a',
    padding='same', trainable=trainable)(input_tensor)
  x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base+'2a')(x)
  x = keras.layers.ReLU()(x)
  
  x = keras.layers.Conv2D(filter2, kernel_size, name=conv_name_base+'2b',
    padding='same', trainable=trainable)(x)
  x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base+'2b')(x)
  x = keras.layers.ReLU()(x)

  x = keras.layers.Conv2D(filter3, kernel_size, name=conv_name_base+'2c',
    padding='same', trainable=trainable)(x)
  x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base+'2c')(x)

  x = keras.layers.Add()([x, input_tensor])
  x = keras.layers.ReLU()(x)
  return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):
  filter1, filter2, filter3 = filters
  bn_axis = 1
  conv_name_base = 'res%s_branch' % (str(stage)+block)
  bn_name_base = 'bn%s_branch' % (str(stage)+block)

  x = keras.layers.Conv2D(filter1, (1, 1), strides=strides, name=conv_name_base+'2a',
    padding='same', trainable=trainable)(input_tensor)
  x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base+'2a')(x)
  x = keras.layers.ReLU()(x)
  
  x = keras.layers.Conv2D(filter2, kernel_size, name=conv_name_base+'2b',
    padding='same', trainable=trainable)(x)
  x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base+'2b')(x)
  x = keras.layers.ReLU()(x)

  x = keras.layers.Conv2D(filter3, kernel_size, name=conv_name_base+'2c',
    padding='same', trainable=trainable)(x)
  x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base+'2c')(x)

  shortcut = keras.layers.Conv2D(filter3, (1, 1), strides=strides, name=conv_name_base + '1',
    padding='same', trainable=trainable)(input_tensor)
  shortcut = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base+'1')(shortcut)

  x = keras.layers.Add()([x, shortcut])
  x = keras.layers.ReLU()(x)
  return x

def nn_base(input_tensor, trainable=True):
  if input_tensor is None:
    img_input = keras.layers.Input(shape=(3, None, None))
  else:
    img_input = keras.layers.Input(tensor=input_tensor, shape=(3, None, None))
  
  bn_axis=1

  # conv1
  x = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same',
    name='conv1', trainable=trainable)(img_input)
  x = keras.layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
  x = keras.layers.ReLU()(x)
  x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

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

def classifier(roi_pool, n_classes=21, trainable=True):
  # roi pool: output from roi pooling
  x = conv_block(roi_pool, 3, [512, 512, 2048], stage=5, block='a', trainable=trainable)
  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', trainable=trainable)
  
  out = keras.layers.AveragePooling2D((7, 7), name='avg_pool')(x)
  out = keras.layers.Flatten()(out)

  out_class = keras.layers.Dense(n_classes, activation='softmax', kernel_initializer='zero',
    name='dense_class_'+n_classes)(out)
  # no regerssion for bg class
  out_regress = keras.layers.Dense(4*(n_classes-1), activation='linear', kernel_initializer='zero',
    name='dense_regress_'+n_classes)(out)
  return [out_class, out_regress]
