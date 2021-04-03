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
vad = stream.WebrtcVADetector()

reader = stream.StreamReader(
        waveFile = waveFile,
        chunkSize = 480,
        simulate = True,
        vaDetector = vad,
      )

# 2. Send packets to remote host
sender = transmit.PacketSender(
        thost = rHostIP,
        tport = rHostPort,
        batchSize = 100,
      )

sender.encode_function = transmit.encode_value_packet

# 3. Receive packets
receiver = transmit.PacketReceiver(bport=bHostPort)

receiver.decode_function = transmit.decode_text_packet

##########################
# Link components
##########################

chain = base.Chain()
chain.add(reader)
chain.add(sender)
chain.add(receiver)

##########################
# Run and display the results
##########################

base.dynamic_run(chain)