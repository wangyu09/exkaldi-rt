#from exkaldirt import base
import base
import numpy as np
import threading
import copy

#########################################
# exkaldirt.base.info
# is an object which has some configs of ExKaldi-RT
#########################################

def test_info():
  
  # Version info
  print( base.info.VERSION )

  # Kaldi root directory if existed
  print( base.info.KALDI_ROOT )

  # Exkaldi-RT compiled c++ root directory if existed
  print( base.info.CMDROOT )

  # The global timeout and timescale threshold (seconds)
  print( base.info.TIMEOUT, base.info.TIMESCALE )

  # The global minimum float value to avoid 0 when compute log
  print( base.info.EPSILON )

  # The maximum resend times using remote connection
  print( base.info.SOCKET_RETRY )

  # The maximum chunk size using remote connection
  print( base.info.MAX_SOCKET_BUFFER_SIZE )

  # reset timeout, timescale, chunk size
  base.info.set_TIMEOUT( 60 )
  base.info.set_TIMEOUT( 0.001 )
  print( base.info.TIMEOUT, base.info.TIMESCALE )

  base.info.set_MAX_SOCKET_BUFFER_SIZE( 65536 )
  print( base.info.MAX_SOCKET_BUFFER_SIZE )

#test_info()

#########################################
# exkaldirt.base.Packet
# is a container to hold stream data.
# We only support 4 types of data: 
# 1: element(int or float value), like 1, 2.0
# 2: vector(numpy 1-d array), like array([1,2,3],dtype="int16")
# 3: matrix(numpy 2-d array), like array([[1.0,2.0],[3.0,4.0]],dtype="float32")
# 4: string(str object), like "hello world"
#
# A packet can carries multiple data.
##########################################

def test_packet():
  
  # Define a packet which carry one frame of mfcc features.
  # Each packet must have a unique chunk id, which is made by a component or joint.
  packet = base.Packet( items={"mfcc":np.ones([13,],dtype="float32") }, cid=0, idmaker=0  )

  # Each packet has a mainKey and you can check all keys in this packet. 
  # This main key is decided automatically when you initialize packet object. 
  print( packet.keys(), packet.mainKey )

  # It has some similar function with Python doct object.
  print( packet.values() )
  print( packet.items() )
  print( packet["mfcc"] ) # but __setitems__ is unavaliable.

  # If you want add a new record in this packet.
  # You can set it as main key.
  packet.add( key="fbank", data=np.ones([24,],dtype="float32"), asMainKey=True )
  print( packet.keys(), packet.mainKey )

  # Packet object can be convert to bytes object for, such as remote transmission.
  bpack = packet.encode()
  print( bpack )

  # A packet can be restored from the corrsponded bytes object.
  npack = base.Packet.decode( bpack )
  print( npack )
  print( npack.keys() )

  # Endpoint is a special packet the mark speech endpoint.
  # It defaultly is an empty packet but also can carries data.
  ENDPOINT = base.Endpoint( cid=0, idmaker=0 )
  bENDPOINT = ENDPOINT.encode()
  print(bENDPOINT)
  nENDPOINT = base.Packet.decode( bENDPOINT )
  print(nENDPOINT)

#test_packet()

#########################################
# exkaldirt.base.PIPE
# is used to pass data and state information between components and joints.
#########################################

def test_pipe():

  # Define a PIPE and put a packet int it.
  # pipe has 5 states:
  # 0 -> silent, 1 -> active, 2 -> terminated, 3 -> wrong, 4 -> stranded (more marks in exkaldirt.base.mark)
  # Different states can allow different operations,
  # for examples, you can not add a new packet into a terminated PIPE.
  pipe = base.PIPE()
  print( pipe.state )
  pipe.put( base.Packet( items={"stream":100}, cid=0, idmaker=0 ) )
  print( pipe.size() )

  # If the input port or/and output port is locked,
  # password is necessary if others want to access it.
  # password is a random int number.
  password = pipe.lock_in()
  print( password )
  print( pipe.is_inlocked() )
  pipe.put( base.Packet( items={"stream":101}, cid=1, idmaker=0 ), password=password )
  print( pipe.size() )

  # When a pipe is locked by a specified component, 
  # you can not get packet from it unless you know the password.
  # you can release it.
  pipe.release_in(password=password)

  # Note that, we will forcely remove continuous endpoints.
  # If the endpoint had data, it will be lost.
  # No matter how many endpoints you added, only keep the first one.
  pipe.put( base.Endpoint( cid=2, idmaker=0 ) )
  print( pipe.size() )
  pipe.put( base.Endpoint( cid=3, idmaker=0 ) )
  print( pipe.size() )
  pipe.put( base.Endpoint( cid=4, idmaker=0 ) )
  print( pipe.size() )

  # For an active pipe, you can:
  # pause it: the state will become "stranded";
  # stop it: the state will become "terminated", and an endpoint packet will be appended at the last automatically;
  # kill it: the state will become "wrong".

  pipe.stop()
  print( pipe.state )
  
  # PIPE is actually a LILO queue.
  # You can get a packet from head.
  print( pipe.get() )

  # If the pipe is "wrong" or "terminated", it can be converted to lists devided by endpoints.
  # you can design the convert rule.
  # defaultly, it only get the main key.
  result = pipe.to_list()
  print(result)

  # If the pipe is "wrong" or "terminated", it can be reset.
  pipe.reset()

  # PIPE can add callback functions.
  # when a packet is appended in PIPE, these functions will work.
  def call_func_1(pac):
    print("feature shape:", pac[pac.mainKey].shape )
  def call_func_2(pac):
    print("feature dtype:", pac[pac.mainKey].dtype )

  pipe.callback( call_func_1 )
  pipe.callback( call_func_2 )

  for i in range(5):
    pipe.put( base.Packet({"data":np.ones([5,],dtype="float32")},cid=i,idmaker=0) )

  # The time info will be recorded when packets are appended and picked out.
  pipe.get()
  print( pipe.report_time() )

#test_pipe()

def test_pipe_communication():

  def PF_func1(pipe):
    for i in range(5):
      pipe.put( base.Packet({"data":1},cid=i,idmaker=0) )
    pipe.stop()

  def PF_func2(pipe):
    while True:
      if pipe.state_is_(base.mark.terminated) and pipe.is_empty():
        break
      else:
        pac = pipe.get()
        print( pac )

  pipe = base.PIPE()
  p1 = threading.Thread(target=PF_func1,args=(pipe,))
  p2 = threading.Thread(target=PF_func2,args=(pipe,))

  p1.start()
  p2.start()

  p1.join()
  p2.join()

#test_pipe_communication()

#########################################
# exkaldirt.base.Component
# is a node to process packets.
#########################################

#########################################
# exkaldirt.base.Joint
# is a node to connect pipeline parallelly (merge or separate).
# Differing from Component, Joint allows multiple inputs and outputs.
# There are some available joints in exkaldi.joint module.
#########################################

def test_joint():

  # Define a joint: it is used to separate different data.
  def combine(items):
    # items is a list including multiple input items (dict objects)
    return {"mfcc":items[0]["mfcc"]}, {"fbank":items[0]["fbank"]}

  joint = base.Joint(jointFunc=combine,outNums=2)

  # Define an input PIPE which the joint will get packets from it.

  pipe = base.PIPE()
  for i in range(5):
    packet = base.Packet( items={"mfcc":np.ones([5,],dtype="float32"),
                                 "fbank":np.ones([4,],dtype="float32")
                                },
                          cid=i,
                          idmaker=0
                        )
    pipe.put( packet )
  pipe.stop()

  # then link joint with this input PIPE
  joint.link(inPIPE=pipe)

  # start and wait
  joint.start()
  joint.wait()

  # there should be two output PIPEs
  print( joint.outPIPE )

  # the first PIPE should only include "mfcc" date
  # the second PIPE should only include "fbank" date
  print( joint.outPIPE[0].get().items() )
  print( joint.outPIPE[1].get().items() )

#test_joint()

#########################################
# exkaldirt.base.Chain
# is used to link and pipeline(s) efficiently
#########################################

def test_chain():

  # Define a component
  class MyComponent(base.Component):

    def __init__(self,*args,**kwargs):
      super().__init__(*args,**kwargs)

    def core_loop(self):
      while True:
        if self.inPIPE.is_empty():
          break
        pack = self.get_packet()
        self.put_packet(pack)
  component = MyComponent()

  # Define a joint
  def copy_packet(items):
    return items[0], copy.deepcopy(items[0])
  
  joint = base.Joint(copy_packet, outNums=2)

  # define the input PIPE
  pipe = base.PIPE()
  for i in range(5):
    pipe.put( base.Packet( {"mfcc":np.ones([5,],dtype="float32") },cid=i, idmaker=0) )
  pipe.stop()

  # Define a chain (container)
  chain = base.Chain()
  # Add a component 
  chain.add( component, inPIPE=pipe )
  # Add a joint. chain will link the joint to the output of component automatically.
  chain.add( joint )

  # Start and Wait
  chain.start()
  chain.wait()

  # Chain has similiar API with component and joint
  print( chain.outPIPE )
  print( chain.outPIPE[0].size() )
  print( chain.outPIPE[1].size() )

test_chain()
