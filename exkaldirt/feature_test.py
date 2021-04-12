import base
import stream
import feature
import time
import numpy as np

wavPath = "../test/84-121550-0000.wav"

####################
# Test feature functions 
####################

def test_functions():

  wavData = stream.read(wavPath).value
  frames = stream.cut_frames(wavData)
  print(frames.shape)

  frames = feature.pre_emphasize_2d(frames)
  print(frames.shape)

  fftLen, frames = feature.split_radix_real_fft_2d(frames)
  print(frames.shape)

#test_functions()

####################
# Test feature extractor
####################

def test_extractor():

  wavData = stream.read(wavPath).value
  frames = stream.cut_frames(wavData)

  pipe = base.PIPE() 

  for i in range(10):
    pipe.put( base.Packet( {"frames":frames[i*50:(i+1)*50]}, cid=i, idmaker=0 ) )

  pipe.stop()
  extractor = feature.MfccExtractor(minParallelSize=1000, oKey="mfcc")

  extractor.start(inPIPE=pipe)
  extractor.wait()

  print( extractor.outPIPE.size() )
  print( extractor.outPIPE.get()["mfcc"].shape )

#test_extractor()

####################
# Test feature processor
####################

def test_processor():

  wavData = stream.read(wavPath).value
  frames = stream.cut_frames(wavData)

  pipe = base.PIPE() 

  for i in range(10):
    pipe.put( base.Packet( {"data":frames[i*50:(i+1)*50]}, cid=i, idmaker=0 ) )

  pipe.stop()

  extractor = feature.MfccExtractor(minParallelSize=1000)
  processor = feature.MatrixFeatureProcessor(spliceLeft=3,spliceRight=3)

  extractor.start(inPIPE=pipe)
  processor.start(inPIPE=extractor.outPIPE)
  processor.wait()

  print( processor.outPIPE.size() )
  packet = processor.outPIPE.get()
  print( packet.keys() )

test_processor()