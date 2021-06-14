import base
import stream
import feature
import joint
import time
import numpy as np

wavPath = "../examples/84-121550-0000.wav"

####################
# exkaldirt.feature.SpectrogramExtractor
# exkaldirt.feature.FbankExtractor
# exkaldirt.feature.MfccExtractor
# are used to extract acoustic feature
####################

def test_spectrogram_extractor():

  # get wave data
  wavData = stream.read(wavPath).value
  frames = stream.cut_frames(wavData)

  # define an input pipe
  pipe = base.PIPE() 
  for i in range(10):
    pipe.put( base.Packet( {"rawWave":frames[i*50:(i+1)*50]}, cid=i, idmaker=0 ) )

  pipe.stop()
  print( pipe.size() )

  # run a spectrogram extractor
  extractor = feature.SpectrogramExtractor(oKey="spectrogram")

  extractor.start(inPIPE=pipe)
  extractor.wait()

  print( extractor.outPIPE.size() )
  packet = extractor.outPIPE.get()
  print( packet.keys() )
  print( packet.mainKey )
  print( packet["spectrogram"].shape )

#test_spectrogram_extractor()

def test_fbank_extractor():

  # get wave data
  wavData = stream.read(wavPath).value
  frames = stream.cut_frames(wavData)

  # define an input pipe
  pipe = base.PIPE() 
  for i in range(10):
    pipe.put( base.Packet( {"rawWave":frames[i*50:(i+1)*50]}, cid=i, idmaker=0 ) )

  pipe.stop()
  print( pipe.size() )

  # run a fbank extractor
  extractor = feature.FbankExtractor(oKey="fbank")

  extractor.start(inPIPE=pipe)
  extractor.wait()

  print( extractor.outPIPE.size() )
  packet = extractor.outPIPE.get()
  print( packet.keys() )
  print( packet.mainKey )
  print( packet["fbank"].shape )

#test_fbank_extractor()

def test_mfcc_extractor():

  # get wave data
  wavData = stream.read(wavPath).value
  frames = stream.cut_frames(wavData)

  # define an input pipe
  pipe = base.PIPE() 
  for i in range(10):
    pipe.put( base.Packet( {"rawWave":frames[i*50:(i+1)*50]}, cid=i, idmaker=0 ) )

  pipe.stop()
  print( pipe.size() )

  # run a mfcc extractor
  extractor = feature.MfccExtractor(oKey="mfcc")

  extractor.start(inPIPE=pipe)
  extractor.wait()

  print( extractor.outPIPE.size() )
  packet = extractor.outPIPE.get()
  print( packet.keys() )
  print( packet.mainKey )
  print( packet["mfcc"].shape )

#test_mfcc_extractor()

def test_mixture_extractor():

  # get wave data
  wavData = stream.read(wavPath).value
  frames = stream.cut_frames(wavData)

  # define an input pipe
  pipe = base.PIPE() 
  for i in range(10):
    pipe.put( base.Packet( {"rawWave":frames[i*50:(i+1)*50]}, cid=i, idmaker=0 ) )

  pipe.stop()
  print( pipe.size() )

  # run a mfcc extractor
  extractor = feature.MixtureExtractor(
                                mixType=["fbank","mfcc"],
                              )

  extractor.start(inPIPE=pipe)
  extractor.wait()

  print( extractor.outPIPE.size() )
  packet = extractor.outPIPE.get()
  print( packet.keys() )
  print( packet.mainKey )
  print( packet["mfcc"].shape )

#test_mixture_extractor()

####################
# exkaldirt.feature.MatrixFeatureProcessor
# is used to transform feature
####################

def test_processor():

  wavData = stream.read(wavPath).value
  frames = stream.cut_frames(wavData)

  pipe = base.PIPE() 

  for i in range(10):
    pipe.put( base.Packet( {"rawWave":frames[i*50:(i+1)*50]}, cid=i, idmaker=0 ) )

  pipe.stop()

  extractor = feature.MfccExtractor(minParallelSize=100,oKey="mfcc")
  processor = feature.MatrixFeatureProcessor(delta=2,spliceLeft=3,spliceRight=3,oKey="mfcc")

  extractor.start(inPIPE=pipe)
  processor.start(inPIPE=extractor.outPIPE, iKey="mfcc")
  processor.wait()

  print( processor.outPIPE.size() )
  packet = processor.outPIPE.get()
  print( packet.keys() )
  print( packet.mainKey )
  print( packet["mfcc"].shape ) # 273 = 13 * 3 * 7

#test_processor()

def test_processor_cmvn():

  wavData = stream.read(wavPath).value
  frames = stream.cut_frames(wavData)

  pipe = base.PIPE() 

  for i in range(10):
    pipe.put( base.Packet( {"rawWave":frames[i*50:(i+1)*50]}, cid=i, idmaker=0 ) )

  pipe.stop()

  extractor = feature.MfccExtractor(minParallelSize=100,oKey="mfcc")
  processor = feature.MatrixFeatureProcessor(
                                    spliceLeft=3,
                                    spliceRight=3,
                                    cmvNormalizer=feature.FrameSlideCMVNormalizer(),
                                    oKey="mfcc",
                                  )

  extractor.start(inPIPE=pipe)
  processor.start(inPIPE=extractor.outPIPE,iKey="mfcc")
  processor.wait()

  print( processor.outPIPE.size() )
  packet = processor.outPIPE.get()
  print( packet.keys() )
  print( packet.mainKey )
  print( packet["mfcc"].shape ) # 273 = 13 * 3 * 7

#test_processor_cmvn()

####################
# There are two ways to extract and process mixture feature.
# The fisrt one, is contruct 
####################

def test_mixture_feature_series():

  wavData = stream.read(wavPath).value
  frames = stream.cut_frames(wavData)

  pipe = base.PIPE() 

  for i in range(10):
    pipe.put( base.Packet( {"rawWave":frames[i*50:(i+1)*50]}, cid=i, idmaker=0 ) )

  pipe.stop()

  extractor = feature.MixtureExtractor(
                              mixType=["fbank","mfcc"],
                            )
  # use a processor to transform fbank feature
  processor1 = feature.MatrixFeatureProcessor(
                                    spliceLeft=2,
                                    spliceRight=2,
                                    cmvNormalizer=feature.FrameSlideCMVNormalizer(),
                                    oKey="fbank",
                                  )
  # use a processor to transform mfcc feature
  processor2 = feature.MatrixFeatureProcessor(
                                    spliceLeft=3,
                                    spliceRight=3,
                                    cmvNormalizer=feature.FrameSlideCMVNormalizer(),
                                    oKey="mfcc",
                                  )

  extractor.start(inPIPE=pipe)
  processor1.start(inPIPE=extractor.outPIPE,iKey="fbank") # specify which key you want to process
  processor2.start(inPIPE=processor1.outPIPE,iKey="mfcc") # specify which key you want to process
  processor2.wait()

  print( processor2.outPIPE.size() )
  packet = processor2.outPIPE.get()
  print( packet.keys() )
  print( packet["fbank"].shape ) # 120 = 24 * 5
  print( packet["mfcc"].shape ) # 91 = 13 * 7

#test_mixture_feature_series()

def test_mixture_feature_parallel():

  wavData = stream.read(wavPath).value
  frames = stream.cut_frames(wavData)

  pipe = base.PIPE() 

  for i in range(10):
    pipe.put( base.Packet( {"rawWave":frames[i*50:(i+1)*50]}, cid=i, idmaker=0 ) )

  pipe.stop()

  extractor = feature.MixtureExtractor(
                              mixType=["fbank","mfcc"],
                            )

  # Split packets
  def split_rule(items):
    return {"fbank":items["fbank"]}, {"mfcc":items["mfcc"]}
  spliter = joint.Spliter(split_rule,outNums=2)
              
  # use a processor to transform fbank feature
  processor1 = feature.MatrixFeatureProcessor(
                                    spliceLeft=2,
                                    spliceRight=2,
                                    cmvNormalizer=feature.FrameSlideCMVNormalizer(),
                                    oKey="fbank",
                                  )
  # use a processor to transform mfcc feature
  processor2 = feature.MatrixFeatureProcessor(
                                    spliceLeft=3,
                                    spliceRight=3,
                                    cmvNormalizer=feature.FrameSlideCMVNormalizer(),
                                    oKey="mfcc",
                                  )

  # combine packets
  def combine_rule(items):
    return { "feat":np.concatenate( [items[0]["fbank"],items[1]["mfcc"]], axis=1) }
  combiner = joint.Combiner(combine_rule)

  extractor.start(inPIPE=pipe)
  spliter.start(inPIPE=extractor.outPIPE)
  processor1.start(inPIPE=spliter.outPIPE[0]) # specify which key you want to process
  processor2.start(inPIPE=spliter.outPIPE[1]) # specify which key you want to process
  combiner.start(inPIPE=[processor1.outPIPE,processor2.outPIPE])
  combiner.wait()

  print( combiner.outPIPE[0].size() )
  packet = combiner.outPIPE[0].get()
  print( packet.keys() )
  print( packet["feat"].shape ) # 211 = 120 + 91

test_mixture_feature_parallel()