from exkaldirt import base,transmit
import os

#################################
# Test sending value packets
#################################

def receive_value_packets():

  receiver = transmit.PacketReceiver(
            bport = 9509,
          )

  receiver.decode_function = transmit.decode_value_packet

  receiver.start()
  receiver.wait()
  print(receiver.outPIPE.size())

#receive_value_packets()

#################################
# Test sending text packets
#################################

def receive_text_packets():

  receiver = transmit.PacketReceiver(
            bport = 9509,
          )

  receiver.decode_function = transmit.decode_text_packet

  #receiver.start()
  #receiver.wait()
  #print(receiver.outPIPE.size())
  base.dynamic_run(receiver)

receive_text_packets()