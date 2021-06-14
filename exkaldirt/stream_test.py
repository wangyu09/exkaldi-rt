#from exkaldirt import stream
import stream
import base
import os
import time

wavPath = "../examples/84-121550-0000.wav"

def test_functions():

  # Read wave info and data
  wav = stream.read(wavPath)
  print(wav.value)

  # Cut the stream into N frames (discard the rest)
  frames1 = stream.cut_frames(wav.value[:-10], width=400, shift=160, snip=True)
  print(frames1.shape)

  # Cut the stream into N frames (retain the rest)
  frames2 = stream.cut_frames(wav.value[:-10], width=400, shift=160, snip=False)
  print(frames2.shape)

####################
# exkaldirt.stream.StreamReader
# is a component used to read real-time stream from file.
####################

def test_stream_reader():

  # Define a stream reader
  reader = stream.StreamReader(
          waveFile = wavPath,
          chunkSize = 480,
          simulate = False,
          #oKey="data",
        )

  reader.start()
  reader.wait()

  # Get the output PIPE and packet
  print( reader.outPIPE.size() )
  pac = reader.outPIPE.get()
  print( pac.mainKey )
  print( pac.keys() )
  print( pac[pac.mainKey] )

#test_stream_reader()

def test_stream_reader_vad():

  # Define a stream reader
  # The webrtc VAD is used 
  reader = stream.StreamReader(
          waveFile = wavPath,
          chunkSize = 480,
          simulate = False,
          vaDetector=stream.WebrtcVADetector(), 
        )

  reader.start()
  reader.wait()

  # Get the output PIPE and packet
  print( reader.outPIPE.size() )
  pac = reader.outPIPE.get()
  print( pac.mainKey )
  print( pac.keys() )
  print( pac[pac.mainKey] )

#test_stream_reader_vad()

####################
# exkaldirt.stream.ElementFrameCutter
# is used to cut real-time stream into frames (or batch frames)
####################

def cutter_test():

  # Define a stream reader
  reader = stream.StreamReader(
          waveFile = wavPath,
          chunkSize = 480,
          simulate = False,
        )

  # Define a cutter
  # if batch size is 1, the output will be a vector (one frame)
  # otherwise, the output will be a matrix (a batch frames)
  cutter = stream.ElementFrameCutter(
          batchSize = 1,
          width = 400,
          shift = 160,
        )

  # Start
  reader.start()
  cutter.start(inPIPE=reader.outPIPE)

  cutter.wait()

  print( cutter.outPIPE.size() )

  pac = cutter.outPIPE.get()
  print( pac.keys() )
  print( pac[pac.mainKey] )

#cutter_test()

####################
# exkaldirt.stream.VectorBatcher
# is used to batch vectors to a matrix
####################

def batcher_test():

  reader = stream.StreamReader(
          waveFile = wavPath,
          chunkSize = 480,
          simulate = False,
        )

  cutter = stream.ElementFrameCutter(
          batchSize = 1,
          width = 400,
          shift = 160,
        )
  
  batcher = stream.VectorBatcher(
          center = 50,
        )

  chain = base.Chain()
  chain.add( reader )
  chain.add( cutter )
  chain.add( batcher )

  chain.start()
  chain.wait()

  print( chain.outPIPE.size() )

#batcher_test()

####################
# exkaldirt.stream.MatrixSubsetter
# is used to split a matrix into N chunks
####################

def subsetter_test():

  reader = stream.StreamReader(
          waveFile = wavPath,
          chunkSize = 480,
          simulate = False,
        )

  cutter = stream.ElementFrameCutter(
          batchSize = 50,
          width = 400,
          shift = 160,
        )
  
  subsetter = stream.MatrixSubsetter(
          nChunk = 2,
        )

  chain = base.Chain()
  chain.add( reader )
  chain.add( cutter )
  chain.add( subsetter )

  chain.start()
  chain.wait()

  print( chain.outPIPE.size() )

#subsetter_test()

####################
# exkaldirt.stream.VectorVADetector
# is used to do VAD
####################

def detector_test():

  reader = stream.StreamReader(
          waveFile = wavPath,
          chunkSize = 480,
          simulate = False,
        )

  cutter = stream.ElementFrameCutter(
          batchSize = 1,
          width = 400,
          shift = 160,
        )
  
  detector = stream.VectorVADetector(
          batchSize=50,
          vadFunc=lambda x:True
        )

  chain = base.Chain()
  chain.add( reader )
  chain.add( cutter )
  chain.add( detector )

  chain.start()
  chain.wait()

  print( chain.outPIPE.size() )

#detector_test()

####################
# exkaldirt.stream.StreamRecorder
# is used to read real-time stream from microphone.
####################

def stream_recorder_test():

  recorder = stream.StreamRecorder()
  recorder.start()

  time.sleep(2)
  recorder.stop()

  recorder.wait()

  print( recorder.outPIPE.size() )

#stream_recorder_test()

def stream_recorder_cutter_test():

  recorder = stream.StreamRecorder(oKey="stream")
  cutter = stream.ElementFrameCutter(batchSize=50,width=400,shift=160,oKey="frames")

  cutter.link(inPIPE=recorder.outPIPE,iKey="stream")

  recorder.start()
  cutter.start()

  time.sleep(2)
  recorder.stop()

  cutter.wait()

  print( cutter.outPIPE.size() )
  print( cutter.outPIPE.get().keys() )

#stream_recorder_cutter_test()

def stream_recorder_cutter_chain_test():

  recorder = stream.StreamRecorder(oKey="stream")
  cutter = stream.ElementFrameCutter(batchSize=50,width=400,shift=160,oKey="frames")

  chain = base.Chain()
  chain.add( node=recorder )
  chain.add( cutter, iKey="stream" )

  chain.start()
  time.sleep(2)
  chain.stop()
  chain.wait()

  print( "size:", chain.outPIPE.size() )

#stream_recorder_cutter_chain_test()