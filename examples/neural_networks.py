import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
  #tf.config.experimental.set_visible_devices(gpu, 'GPU')

def make_DNN_acoustic_model(inputDim, outDimPdf):
  
  model = keras.models.Sequential()
  for i in range(5):
    if i == 0:
      model.add( keras.layers.Dense(1024, activation=None, use_bias=False, input_shape=(inputDim,)) )
    else:
      model.add( keras.layers.Dense(1024, activation=None, use_bias=False, input_shape=(1024,)) )
    model.add( keras.layers.BatchNormalization(momentum=0.95) )
    model.add( keras.layers.ReLU() )
    model.add( keras.layers.Dropout(0.2) )

  model.add( keras.layers.Dense(outDimPdf,
                                activation=None,
                                use_bias=True,
                                kernel_initializer="he_normal", 
                                bias_initializer='zeros',
                                input_shape=(1024,)
                                ) 
                              )

  return model

def load_denoise_model( modelPath ):
  return tf.keras.models.load_model( modelPath )