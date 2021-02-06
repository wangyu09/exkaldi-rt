from exkaldi2 import base,stream,feature,decode

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras

##########################
# Step 1: Create all components
##########################

# 1. Create a stream reader to read realtime stream from audio file
reader = stream.StreamReader("1088-134315-0000.wav",False)
# 2. Cutter to cut frame
cutter = stream.FrameCutter(width=400,shift=160)
# 3. fbank feature extracting
extractor = feature.FbankExtractor(
                          frameDim = 400, 
                          batchSize = 16,
                          dither=0.0,
                          useEnergy=False,
                        )
# 4. processing feature
processor = feature.FeatureProcessor(
                          rawFeatDim=23,
                          batchSize=20,
                          delta=0,
                          spliceLeft=2,
                          spliceRight=2,
                        )
# 5. acoustic probability computer
acousticmodel = decode.AcousticEstimator(
                          featDim=115,
                          batchSize=20,
                          padFinal=False,
                          applySoftmax=True,
                          applyLog=True,
                          priors=None
                        )

def make_DNN_model(inputDim,outDimPdf):
    
  inputs = keras.Input((inputDim,))

  ln1 = keras.layers.Dense(1024, activation=None, use_bias=False)(inputs)
  ln1_bn = keras.layers.BatchNormalization(momentum=0.95)(ln1)
  ln1_ac = keras.layers.ReLU()(ln1_bn)
  ln1_do = keras.layers.Dropout(0.2)(ln1_ac)

  ln2 = keras.layers.Dense(1024, activation=None, use_bias=False)(ln1_do)
  ln2_bn = keras.layers.BatchNormalization(momentum=0.95)(ln2)
  ln2_ac = keras.layers.ReLU()(ln2_bn)
  ln2_do = keras.layers.Dropout(0.2)(ln2_ac)

  ln3 = keras.layers.Dense(1024, activation=None, use_bias=False)(ln2_do)
  ln3_bn = keras.layers.BatchNormalization(momentum=0.95)(ln3)
  ln3_ac = keras.layers.ReLU()(ln3_bn)
  ln3_do = keras.layers.Dropout(0.2)(ln3_ac)

  ln4 = keras.layers.Dense(1024, activation=None, use_bias=False)(ln3_do)
  ln4_bn = keras.layers.BatchNormalization(momentum=0.95)(ln4)
  ln4_ac = keras.layers.ReLU()(ln4_bn)
  ln4_do = keras.layers.Dropout(0.2)(ln4_ac)

  ln5 = keras.layers.Dense(1024, activation=None, use_bias=False)(ln4_do)
  ln5_bn = keras.layers.BatchNormalization(momentum=0.95)(ln5)
  ln5_ac = keras.layers.ReLU()(ln5_bn)
  ln5_do = keras.layers.Dropout(0.2)(ln5_ac)

  outputs_pdf = keras.layers.Dense(outDimPdf,activation=None,use_bias=True,kernel_initializer="he_normal", bias_initializer='zeros', name="pdfID")(ln5_do)

  #outputs_pho = keras.layers.Dense(outDimPho,activation=None,use_bias=True,kernel_initializer="he_normal", bias_initializer='zeros', name="phoneID")(ln5_do)

  return keras.Model(inputs=inputs, outputs=outputs_pdf)

kerasmodel = make_DNN_model(115,2016)
kerasmodel.load_weights("/misc/Work19/wangyu/exkaldi-online-test/mini_librispeech/exp/model_ep4.h5")

def keras_compute(feats):
  global kerasmodel
  predPdf = kerasmodel(feats, training=False)
  return predPdf.numpy()

acousticmodel.acoustic_function = keras_compute

# 6. online decoder
decoder = decode.WfstDecoder(
                      probDim=2016,
                      batchSize=20,
                      symbolTable="/misc/Work18/wangyu/kaldi/egs/mini_librispeech/s5/data/lang/words.txt",
                      silencePhones="1:2:3:4:5:6:7:8:9:10",
                      frameShiftSec=0.01,
                      tmodel="/misc/Work18/wangyu/kaldi/egs/mini_librispeech/s5/exp/tri3b/final.mdl",
                      graph="/misc/Work18/wangyu/kaldi/egs/mini_librispeech/s5/exp/tri3b/graph_tgsmall/HCLG.fst",
                      beam=10,
                      latticeBeam=8,
                      minActive=200,
                      maxActive=7000,
                      acousticScale=0.1
                    )

##########################
# Step 2: Link this components
##########################

chain = base.Chain()

chain.add(reader)
chain.add(cutter)
chain.add(extractor)
chain.add(processor)
chain.add(acousticmodel)
chain.add(decoder)

##########################
# Step 3: Start running
##########################

chain.start()
base.wait_and_dynamic_display(chain,items=["is_endpoint","data"])