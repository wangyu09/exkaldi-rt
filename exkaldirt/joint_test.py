import base
import joint
import numpy as np

####################
# exkaldirt.joint.Mapper
# is used to map packets.
# Mapper is actually a component object, 
# because it only has one input and one output.
####################

def test_mapper():

  pipe = base.PIPE()
  for i in range(5):
    pipe.put( base.Packet({"mfcc":np.ones([5,],dtype="float32")},cid=i,idmaker=0) )

  pipe.stop()  

  def map_function(items):
    return { "scaledMFCC": 2*items["mfcc"] }
  
  mapper = joint.Mapper(mapFunc=map_function)
  mapper.start(inPIPE=pipe)
  mapper.wait()

  print( mapper.outPIPE.size() )
  print( mapper.outPIPE.get()["scaledMFCC"] )

#test_mapper()

####################
# exkaldirt.joint.Spliter
# is used to split one packet into N packets.
# Spliter is a Joint object.
####################

def test_spliter():

  pipe = base.PIPE()
  for i in range(5):
    pipe.put( base.Packet( items={"mfcc":np.ones([5,],dtype="float32"),
                                  "fbank":np.ones([5,],dtype="float32"),
                            },
                            cid=i,idmaker=0) 
            )

  pipe.stop()  

  def split_function(items):
    return [ {key:value} for key,value in items.items() ]
  
  spliter = joint.Spliter(func=split_function,outNums=2)
  spliter.start(inPIPE=pipe)
  spliter.wait()

  print( spliter.outPIPE )
  print( spliter.outPIPE[0].get().keys() )
  print( spliter.outPIPE[1].get().keys() )

#test_spliter()

####################
# exkaldirt.joint.Replicator
# is used to copy one packet into N packets.
# replicator is a Joint object.
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
# exkaldirt.joint.Combiner
# is used to combine N packets to one with a specified rule.
# combiner is a Joint object.
####################

def test_combiner():

  pipe1 = base.PIPE()
  pipe2 = base.PIPE()
  for i in range(5):
    pipe1.put( base.Packet({"mfcc":np.ones([5,],dtype="float32")},cid=i,idmaker=0) )
    pipe2.put( base.Packet({"fbank":np.zeros([5,],dtype="float32")},cid=i,idmaker=0) )

  pipe1.stop()
  pipe2.stop()

  def combine_function(items):
    return { "mixedFeat":np.concatenate([items[0]["mfcc"],items[1]["fbank"]]) }

  combiner = joint.Combiner(func=combine_function)

  combiner.link(inPIPE=[pipe1,pipe2])

  combiner.start()
  combiner.wait()

  print( combiner.outPIPE )
  print( combiner.outPIPE[0].get().items() )

#test_combiner()

####################
# exkaldirt.joint.Merger
# is used to combine N packets to one.
# merger is a Joint object.
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

  print( merger.outPIPE )
  print( merger.outPIPE[0].size() )
  print( merger.outPIPE[0].get().items() )

#test_merger()