from exkaldirt import stream, transmit
import time
import wave
import numpy as np

##########################
# Hyperparameters
##########################

rHostIP = "192.168.1.5"
rHostPort = 9509
bHostPort = 9510

##########################
# Define components
##########################

recorder = stream.StreamRecorder()

cutter = stream.ElementFrameCutter(
                              batchSize=1,
                              width=64,
                              shift=64
                            )

sender = transmit.PacketSender(rHostIP,rHostPort)

##########################
# Link components
##########################

recorder.start()
cutter.start( inPIPE=recorder.outPIPE )
sender.start( inPIPE=cutter.outPIPE )

time.sleep(10)
recorder.stop()
sender.wait()
