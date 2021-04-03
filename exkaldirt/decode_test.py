from exkaldirt import stream,base
from exkaldirt import feature
from exkaldirt import decode
import os
import numpy as np

wavPath = "../examples/84-121550-0000.wav"

assert os.path.isfile(wavPath), f"No such file: {wavPath}"

###########################
# Acoustic Estimator
###########################

def feat_estimator_test():

  reader = stream.StreamReader(
          waveFile = wavPath,
          chunkSize = 480,
          simulate = False,
          vaDetector = None,
        )

  cutter = stream.ElementFrameCutter(
          width = 400,
          shift = 160,
        )
  
  extractor = feature.MfccExtractor(
          batchSize = 100,
          useEnergy = False,
        )

  processor = feature.FeatureProcessor(
          featDim = 13,
          delta = 2,
          spliceLeft = 10,
          spliceRight = 10,
          cmvNormalizer = feature.FrameSlideCMVNormalizer(),
        )
  
  left = 5
  right = 5
  estimator = decode.AcousticEstimator(
          featDim = 819,
          batchSize = 100,
          applySoftmax = False,
          applyLog = False,
          leftContext = left,
          rightContext = right,
        )

  estimator.acoustic_function = lambda x:x[left:-right].copy()

  reader.start()
  cutter.start(inPIPE=reader.outPIPE)
  extractor.start(inPIPE=cutter.outPIPE)
  processor.start(inPIPE=extractor.outPIPE)
  estimator.start(inPIPE=processor.outPIPE)

  estimator.wait()
  print( estimator.outPIPE.size() )

feat_estimator_test()
