from exkaldirt import stream
from exkaldirt import feature
import os

wavPath = "../examples/84-121550-0000.wav"

assert os.path.isfile(wavPath), f"No such file: {wavPath}"

###########################
# Feature Extractor
###########################

def feat_extractor_test():

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

  reader.start()
  cutter.start(inPIPE=reader.outPIPE)
  extractor.start(inPIPE=cutter.outPIPE)

  extractor.wait()
  print( extractor.outPIPE.size() )

#feat_extractor_test()

###########################
# Mixture Feature Extractor
###########################

def mixture_feat_extractor_test():

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
  
  extractor = feature.MixtureExtractor(
          frameDim = 400,
          batchSize = 100,
          mixType = ["mfcc","fbank"],
          useEnergyForFbank = False,
          useEnergyForMfcc = False,
        )

  reader.start()
  cutter.start(inPIPE=reader.outPIPE)
  extractor.start(inPIPE=cutter.outPIPE)

  extractor.wait()
  print( extractor.outPIPE.size() )
  pac = extractor.outPIPE.get()
  print( pac.data.shape )

#mixture_feat_extractor_test()

###########################
# Feature Processor
###########################

def feat_processor_test():

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

  reader.start()
  cutter.start(inPIPE=reader.outPIPE)
  extractor.start(inPIPE=cutter.outPIPE)
  processor.start(inPIPE=extractor.outPIPE)

  processor.wait()
  print( processor.outPIPE.size() )
  pac = processor.outPIPE.get()
  print( pac.data.shape )

feat_processor_test()
