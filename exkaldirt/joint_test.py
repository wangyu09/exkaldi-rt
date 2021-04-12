import base
import joint
import numpy as np

####################
# Test Replicator
####################

def test_replicator():

  replicator = joint.Replicator(outNums=2)

  pipe = base.PIPE()
  for i in range(5):
    pipe.put( base.Packet({"mfcc":np.ones([5,],dtype="float32")},cid=i,idmaker=0) )

  pipe.stop()

  replicator.link(inPIPE=pipe)
  replicator.start()
  replicator.wait()

  print( replicator.outPIPE )

  print( replicator.outPIPE[0].get().items() )
  print( replicator.outPIPE[1].get().items() )

#test_replicator()

####################
# Test Merger
####################

def test_merger():

  merger = joint.Merger()

  pipe1 = base.PIPE()
  pipe2 = base.PIPE()
  for i in range(5):
    pipe1.put( base.Packet({"mfcc":np.ones([5,],dtype="float32")},cid=i,idmaker=0) )
    pipe2.put( base.Packet({"fbank":np.ones([4,],dtype="float32")},cid=i,idmaker=0) )

  pipe1.stop()
  pipe2.stop()

  merger.link(inPIPE=[pipe1,pipe2])
  merger.start()
  merger.wait()

  print( merger.outPIPE[0].size() )

  print( merger.outPIPE[0].get().items() )

#test_merger()