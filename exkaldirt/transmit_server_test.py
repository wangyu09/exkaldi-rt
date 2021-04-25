import base
import transmit
import numpy as np
import socket

#############################
# exkaldirt.transmit.PacketReceiver
# is used to receive packet from remote host computer.
#############################

def test_receiver():

  # define a receiver, bind local host IP and a port
  receiver = transmit.PacketReceiver(bport=9509)

  # run
  receiver.start()
  receiver.wait()

  print( receiver.outPIPE.size() )
  #
  while True:
    if receiver.outPIPE.is_empty():
      break
    packet = receiver.outPIPE.get()
    print( packet.items() )

test_receiver()

