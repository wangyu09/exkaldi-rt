from exkaldirt import base, stream, transmit
import os

#################################
# Test sending value packets
#################################

def send_value_packets():

  wavPath = "../examples/84-121550-0000.wav"

  assert os.path.isfile(wavPath), f"No such file: {wavPath}"

  reader = stream.StreamReader(
          waveFile = wavPath,
          chunkSize = 480,
          simulate = False,
          vaDetector = None,
        )

  sender = transmit.PacketSender(
          thost = "192.168.1.11",
          tport = 9509,
          batchSize = 1024,
        )

  sender.encode_function = transmit.encode_value_packet

  reader.start()
  sender.start( inPIPE=reader.outPIPE )
  sender.wait()

#send_value_packets()

#################################
# Test sending text packets
#################################

def send_text_packets():

  testPIPE = base.PIPE()
  for i in range(1,21):
    testPIPE.put( base.Text( "AA "*i ) )
  testPIPE.stop()

  sender = transmit.PacketSender(
          thost = "192.168.1.11",
          tport = 9509,
          batchSize = 1024,
        )

  sender.encode_function = transmit.encode_text_packet

  sender.start( inPIPE=testPIPE )
  sender.wait()

send_text_packets()