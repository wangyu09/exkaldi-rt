from exkaldirt import stream
import os

wavPath = "../examples/84-121550-0000.wav"

assert os.path.isfile(wavPath), f"No such file: {wavPath}"
#print(stream.read(wavPath))

#print(stream.record(seconds=2))

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

  print(reader.outPIPE.size())

#stream_reader_test()

####################
# Test Cutter
####################

def cutter_test():

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
  
  reader.start()
  cutter.start(inPIPE=reader.outPIPE)

  cutter.wait()
  print( cutter.outPIPE.size() )

#cutter_test()

