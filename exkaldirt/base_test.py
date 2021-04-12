#from exkaldirt import base
import base
import numpy as np
import multiprocessing
import copy

#########################################
# Packet
#########################################

def test_packet():
  packet = base.Packet( items={"mfcc":np.ones([5,],dtype="float32") }, cid=0, idmaker=0  )
  print( packet )
  print( packet.mainKey )
  print( packet.cid )
  print( packet.idmaker )
  print( packet.keys() )
  print( packet["mfcc"] )

  packet.add( "fbank", np.ones([4,3],dtype="float32"), asMainKey=True )
  print( packet.keys() )
  print( packet.mainKey )
  print( packet[packet.mainKey] )

  bpack = packet.encode()
  print( bpack )
  print( len(bpack) )
  npack = base.Packet.decode( bpack )
  print( npack )
  print( npack.keys() )
  print( npack["mfcc"] )
  print( npack["fbank"] )

#########################################
# PIPE
#########################################

def test_pipe():

  pipe = base.PIPE()
  print( pipe.name, pipe.objid, pipe.state, pipe.timestamp )
  password = pipe.lock_in()
  print(password)

  packet = base.Packet( items={"mfcc":np.ones([5,],dtype="float32") }, cid=0, idmaker=0  )
  pipe.put( packet, password=password )
  print( pipe.size() )

  pipe.pause()
  pipe.get()
  pipe.stop()
  print( pipe.to_list() )

  newPipe = base.PIPE()

  def test_fn(pac):
    print("feature shape:", pac[pac.mainKey].shape )

  newPipe.callback( test_fn )

  for i in range(5):
    newPipe.put( base.Packet({"mfcc":np.ones([5,],dtype="float32")},cid=i,idmaker=0) )

  def PF_func1(pipe):
    for i in range(5):
      pipe.put( base.Packet({"mfcc":np.ones([5,],dtype="float32")},cid=0,idmaker=0) )
    pipe.stop()

  def PF_func2(pipe):
    while True:
      if pipe.state_is_(base.mark.terminated) and pipe.is_empty():
        break
      else:
        pac = pipe.get()
        print( pac )

  pipe = base.PIPE()
  p1 = multiprocessing.Process(target=PF_func1,args=(pipe,))
  p2 = multiprocessing.Process(target=PF_func2,args=(pipe,))

  p1.start()
  p2.start()

  p1.join()
  p2.join()

  print( pipe.size() )

test_pipe()

#########################################
# Joint test
#########################################

def test_joint():

  def combine(items):
    return items[1],items[0]

  joint = base.Joint(combine,outNums=2)

  pipe1 = base.PIPE()
  pipe2 = base.PIPE()
  for i in range(5):
    pipe1.put( base.Packet({"mfcc":np.ones([5,],dtype="float32")},cid=i,idmaker=0) )
    pipe2.put( base.Packet({"fbank":np.ones([4,],dtype="float32")},cid=i,idmaker=0) )

  pipe1.stop()
  pipe2.stop()

  joint.link(inPIPE=[pipe1,pipe2])
  joint.start()
  joint.wait()

  print( joint.outPIPE[0].size() )

  print( joint.outPIPE[0].get().items() )
  print( joint.outPIPE[1].get().items() )

#test_joint()

#########################################
# Chain test
#########################################

class MyComponent(base.Component):

  def core_loop(self):
    while True:
      if self.inPIPE.is_empty():
        break
      pack = self.get_packet()
      self.put_packet(pack)

def copy_packet(items):
  return items[0], copy.deepcopy(items[0])

def test_chain():

  pipe = base.PIPE()
  for i in range(5):
    pipe.put( base.Packet( {"mfcc":np.ones([5,],dtype="float32") },cid=i, idmaker=0) )
  pipe.stop()

  component = MyComponent()
  joint = base.Joint(copy_packet, outNums=2)

  chain = base.Chain()
  chain.add( component, inPIPE=pipe )
  chain.add( joint )

  chain.start()
  chain.wait()

  print( chain.outPIPE )
  print( chain.outPIPE[0].size() )
  print( chain.outPIPE[1].size() )

#test_chain()
