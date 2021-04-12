#from exkaldirt import stream
import stream
import base
import os

wavPath = "../test/84-121550-0000.wav"

assert os.path.isfile(wavPath), f"No such file: {wavPath}"
#wav = stream.read(wavPath)
#print(wav.value)

#frames1 = stream.cut_frames(wav.value[:-10])
#print(frames1.shape)

#frames2 = stream.cut_frames(wav.value[:-10],snip=False)
#print(frames2.shape)

####################
# Test Stream Reader
####################

def stream_reader_test():

  vad = None #stream.WebrtcVADetector()

  reader = stream.StreamReader(
          waveFile = wavPath,
          chunkSize = 480,
          simulate = False,
          vaDetector = vad,
        )

  reader.start()
  reader.wait()

  print( reader.outPIPE.size() )
  print( reader.outPIPE.state_is_(base.mark.terminated) )
  print( reader.outPIPE.is_inlocked() )
  print( reader.outPIPE.is_outlocked() )
  print( reader.get_audio_info() )

  pac = reader.outPIPE.get()
  print( pac.mainKey )
  print( pac.keys() )
  print( pac[pac.mainKey] )

#stream_reader_test()

####################
# Test Cutter
####################

def cutter_test():

  reader = stream.StreamReader(
          waveFile = wavPath,
          chunkSize = 480,
          simulate = True,
          vaDetector = None,
        )

  cutter = stream.ElementFrameCutter(
          batchSize = 1,
          width = 400,
          shift = 160,
        )

  reader.start()
  cutter.start(inPIPE=reader.outPIPE)
  #base.dynamic_display( cutter.outPIPE )
  cutter.wait()

  print( cutter.outPIPE.size() )

  pac = cutter.outPIPE.get()
  print( pac.keys() )
  print( pac[pac.mainKey] )

#cutter_test()

####################
# Test Batcher
####################

def batcher_test():

  reader = stream.StreamReader(
          waveFile = wavPath,
          chunkSize = 480,
          simulate = True,
          vaDetector = None,
        )

  cutter = stream.ElementFrameCutter(
          batchSize = 1,
          width = 400,
          shift = 160,
        )
  
  batcher = stream.VectorBatcher(
          center = 50,
        )

  reader.start()
  cutter.start(inPIPE=reader.outPIPE)
  batcher.start(inPIPE=cutter.outPIPE)
  batcher.wait()

  print( batcher.outPIPE.size() )

#batcher_test()

####################
# Test VAD
####################

def detector_test():

  reader = stream.StreamReader(
          waveFile = wavPath,
          chunkSize = 480,
          simulate = True,
          vaDetector = None,
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

  reader.start()
  cutter.start(inPIPE=reader.outPIPE)
  detector.start(inPIPE=cutter.outPIPE)
  detector.wait()

  print( detector.outPIPE.size() )

#detector_test()


