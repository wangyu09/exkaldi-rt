import base
import transmit
import numpy as np

#############################
# exkaldirt.transmit.PacketSender
# is used to send packet to remote host computer.
#############################

def test_sender():

  pipe = base.PIPE()

  for i in range(5):
    pipe.put( base.Packet( {"element":1}, cid=i, idmaker=0 ) )

  for i in range(5,10):
    pipe.put( base.Packet( {"vector":np.ones([5,],dtype="float32")}, cid=i, idmaker=0 ) )

  for i in range(10,15):
    pipe.put( base.Packet( {"matrix":np.ones([5,10],dtype="float32")}, cid=i, idmaker=0 ) )

  for i in range(15,20):
    pipe.put( base.Packet( {"string":"this is a test"}, cid=i, idmaker=0 ) )

  pipe.stop()

  # define a packet sender
  sender = transmit.PacketSender(thost="192.168.1.11",tport=9509)
  sender.start(inPIPE=pipe)
  sender.wait()

test_sender()