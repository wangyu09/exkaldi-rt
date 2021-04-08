#from exkaldirt import base
import base
import numpy as np
import multiprocessing

#########################################
# Packet
#########################################

packet = base.Packet( items={"mfcc":np.ones([5,],dtype="float32") }, sid=0, eid=4, idmaker=0  )
print( packet )
print( packet.mainKey )
print( packet.sid )
print( packet.eid )
print( packet.idmaker )
print( packet.keys() )
print( packet["mfcc"] )

print( packet.update_id(5,8) )
print( packet.sid )
print( packet.eid )

packet.add( "fbank", np.ones([4,],dtype="float32"), asMainKey=True )
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

pipe = base.PIPE()
print( pipe.name, pipe.objid, pipe.state, pipe.timestamp )
password = pipe.lock_in()
print(password)
pipe.put( npack, password=password )
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
  newPipe.put( base.Packet({"mfcc":np.ones([5,],dtype="float32")},sid=i,eid=i,idmaker=0) )

def PF_func1(pipe):

  for i in range(5):
    pipe.put( base.Packet({"mfcc":np.ones([5,],dtype="float32")},sid=i,eid=i,idmaker=0) )

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
