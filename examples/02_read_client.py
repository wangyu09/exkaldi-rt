from exkaldirt import base, stream, transmit
import os

##########################
# Hyperparameters
##########################

waveFile = "84-121550-0000.wav"
rHostIP = "192.168.1.11"
rHostPort = 9509
bHostPort = 9510

assert os.path.isfile(waveFile), f"No such file: {waveFile}"

##########################
# Define components
##########################

# 1. Create a stream reader to read realtime stream from audio file
vad = None #stream.WebrtcVADetector()

reader = stream.StreamReader(
        waveFile = waveFile,
        chunkSize = 480,
        simulate = True,
        vaDetector = vad,
      )

# 2. Send packets to remote host
cutter = stream.ElementFrameCutter(
        batchSize = 1,
        width=64,
        shift=64,
      )

sender = transmit.PacketSender(
        thost = rHostIP,
        tport = rHostPort,
      )

# 3. Receive packets
#receiver = transmit.PacketReceiver(bport=bHostPort)

##########################
# Link components
##########################

chain = base.Chain()
chain.add(reader)
chain.add(cutter)
chain.add(sender)

""" def call_func(pack):
  print("decode out:", pack["data"])
receiver.outPIPE.callback(call_func)
chain.add(receiver)
"""

chain.start()
chain.wait()