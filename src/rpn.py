# region proposal network

import tensorflow
from tensorflow import keras

def rpn(base_layers, num_anchors):
  x = keras.layers.Conv2D(512, 3, name='rpn_conv1', activation='relu',
    padding='same', kernel_initializer='normal')(base_layers)
  x_class = keras.layers.Conv2D(num_anchors, 1, name='rpn_out_class', activation='sigmoid',
    padding='same', kernel_initializer='uniform')(x)
  x_regress = keras.layers.Conv2D(num_anchors*4, 1, name='rpn_out_regress', activation='linear',
    padding='same', kernel_initializer='zero')(x)
  return [x_class, x_regress, base_layers]

