from stream import StreamReader, FrameCutter
from feature import FbankExtractor, FeatureProcessor
from decode import AcousticEstimator, WfstDecoder
import time
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras

reader = StreamReader("1088-134315-0000.wav",binary=False)

cutter = FrameCutter(width=400,shift=160)

extractor = FbankExtractor(
                          frameDim = 400, 
                          chunkFrames = 16,
                          dither=0.0,
                          useEnergy=False,
                        )

processor = FeatureProcessor(
              rawFeatDim=23,
              delta=0,
              spliceLeft=2,
              spliceRight=2,
            )

acousticmodel = AcousticEstimator(
              featDim=115,
              chunkFrames=20,
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
kerasmodel.load_weights("/misc/Work18/wangyu/kaldi/src/mybin/mini_librispeech/exp/model_ep4.h5")

def keras_compute(feats):
  global kerasmodel
  predPdf = kerasmodel(feats, training=False)
  return predPdf.numpy()

acousticmodel.acoustic_function = keras_compute

decoder = WfstDecoder(
                      probDim=2016,
                      chunkFrames=20,
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
f0pp = reader.get_stream_pipe()
f1pp = cutter.get_frame_pipe()
f2pp = extractor.get_feature_pipe()

f3pp = processor.get_feature_pipe()
f4pp = acousticmodel.get_probability_pipe()
f5pp = decoder.get_result_pipe()

reader.start_reading()
st = time.time()
cutter.start_cutting(streamPIPE=f0pp)
extractor.start_extracting(framePIPE=f1pp)
processor.start_processing(rawFeaturePIPE=f2pp)
acousticmodel.start_estimating(featurePIPE=f3pp)
decoder.start_decoding(probabilityPIPE=f4pp)

while not f5pp.is_exhaustion():
  #print("stream:",f0pp.size(),"frame:",f1pp.size(), "rawFeat:",f2pp.size(),"feat:",f3pp.size(),"prob:",f3pp.size())
  if decoder.is_termination() and f5pp.is_empty():
    break
  if not f5pp.is_empty():
    best1 = f5pp.get()
    print()
    print("Endpoint?:", best1.is_endpoint(), "Result:", best1.item)
  else:
    time.sleep(0.01)
print("total time cost:",time.time()-st)

time.sleep(1)
