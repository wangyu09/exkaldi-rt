# coding=utf-8
#
# Yu Wang (University of Yamanashi)
# Feb, 2021
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import queue
import subprocess
import numpy as np
import sys
import threading
import time

from exkaldi2.version import version

## Some configs.
class Info:

  def __init__(self):
    self.__timeout = 10
    self.__timescale = 0.01
    self.__cmdroot = self.__find_cmd_root()
    self.__epsilon = self.__get_floot_floor()

  @property
  def VERSION(self):
    return version

  @property
  def CMDROOT(self):
    return self.__cmdroot

  @property
  def TIMEOUT(self):
    return self.__timeout
  
  @property
  def TIMESCALE(self):
    return self.__timescale
  
  @property
  def EPSILON(self):
    return self.__epsilon

  @property
  def SOCKET_RETRY(self):
    return 10

  def set_TIMEOUT(self,value):
    assert isinstance(value,int) and value > 0, "TIMEOUT must be an int value."
    self.__timeout = value
  
  def set_TIMESCALE(self,value):
    assert isinstance(value,float) and 0 < value < 1.0, "TIMESCALE should be a float value in (0,1)."
    self.__timescale = value

  def __find_cmd_root(self):
    '''Look for the exkaldi-online command root path.'''
    if "KALDI_ROOT" in os.environ.keys():
      KALDI_ROOT = os.environ["KALDI_ROOT"]
    else:
      cmd = "which copy-matrix"
      p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy())
      out, err = p.communicate()
      if out == b'':
        raise Exception("Kaldi root directory was not found automatically. Please ensure it has been added in environment sucessfully.")
      else:
        out = out.decode().strip()
        KALDI_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(out)))

    assert os.path.isfile(os.path.join(KALDI_ROOT,"src","exkaldionlinebin","exkaldi-online-decoder")), \
          "ExKaldi-Online C++ source files have not been compiled sucessfully. " + \
          "Please consult the Installation in github: https://github.com/wangyu09/exkaldi-online ."

    return os.path.join(KALDI_ROOT,"src","exkaldionlinebin")

  def __get_floot_floor(self):
    '''Get the floot floor value'''
    cmd = os.path.join(self.__cmdroot,"get-float-floor")
    p = subprocess.Popen(cmd,stdout=subprocess.PIPE)
    (out,_) = p.communicate()
    return float(out.decode().strip())

info = Info()

## Base class to describe state of components and pipes
class StateFlag:

  SILENT = 0
  ALIVE = 1
  TERMINATION = 2
  ERROR = 3

  def __init__(self):
    self.__state = StateFlag.SILENT

  def is_silent(self):
    return self.__state == StateFlag.SILENT

  def is_alive(self):
    return self.__state == StateFlag.ALIVE
  
  def is_error(self):
    return self.__state == StateFlag.ERROR
  
  def is_termination(self):
    return self.__state == StateFlag.TERMINATION
  
  def shift_state_to_silent(self):
    self.__state = StateFlag.SILENT

  def shift_state_to_alive(self):
    self.__state = StateFlag.ALIVE

  def shift_state_to_error(self):
    self.__state = StateFlag.ERROR

  def shift_state_to_termination(self):
    self.__state = StateFlag.TERMINATION

## PIPE is used to connect components and pass data packets
class PIPE(StateFlag):
  '''
  It is a Last-In-Last-Out queue.
  '''
  def __init__(self):
    super().__init__()
    self.__cache = queue.Queue()
    self.__extra_info = None
    self.shift_state_to_alive() # Defaultly it is activated.
  
  def kill(self):
    self.shift_state_to_error()
    self.__cache.queue.clear()
    self.__extra_info = None
  
  def stop(self):
    self.shift_state_to_termination()
  
  def is_empty(self):
    return self.__cache.empty()

  def is_exhaustion(self):
    return self.is_termination() and self.is_empty()

  def clear(self):
    self.__cache.queue.clear()
  
  def size(self):
    return self.__cache.qsize()
  
  def get(self):
    '''
    Pop a packet from head.
    '''
    if self.is_exhaustion():
      raise Exception("PIPE has terminated and nothing is left.") 
    elif self.is_error():
      raise Exception("Can not get data from a killed PIPE." )
    return self.__cache.get(timeout=info.TIMEOUT)
  
  def put(self,packet):
    '''
    Push a new packet to tail.
    '''
    if not self.is_alive():
      raise Exception("Can only append data into ALIVE PIPE." )
    assert isinstance(packet,Packet), "This is not a Packet object."
    self.__cache.put(packet)
  
  def add_extra_info(self,info=None):
    '''
    Add any extra information to this PIPE.
    '''
    self.__extra_info = info
  
  def get_extra_info(self):
    '''
    Get the extra information added to this PIPE.
    '''
    return self.__extra_info

  def to_list(self,deep=True):
    '''
    Convert PIPE to list.
    
    Args:
      _deep_: If True, return list of values. Or return list of packets.
    '''
    assert isinstance(deep,bool), "<deep> need a bool value."
    size = self.size()
    if deep:
      return [ (self.__cache.get()).data for i in range(size) ]
    else:
      return [ self.__cache.get() for i in range(size) ]

## Packet is used to hold various stream data
## These data will be processed by Component and passed in PIPE
class Packet:

  def __init__(self,data,endpoint=False):
    assert isinstance(endpoint,bool), "<endpoint> need a bool value."
    self.__data = data
    self.__endpoint = endpoint

  @property
  def data(self):
    return self.__data

  def is_endpoint(self):
    return self.__endpoint

  @property
  def dtype(self):
    raise Exception("Please implement this function.")

## Element packet hold singal value data.
class Element(Packet):
  
  def __init__(self,data,endpoint=False):
    if isinstance(data,int):
      data = np.int16(data)
    elif isinstance(data,float):
      data = np.int32(data)
    else:
      assert isinstance(data,(np.int8,np.int16,np.int32,
                             np.float16,np.float32,np.float64)), "Element packet must be int or float value."
    super().__init__(data,endpoint)
  
  @property
  def dtype(self):
    return str(self.data.dtype)

## Vector packet hold 1-d array data.
class Vector(Packet):

  def __init__(self,data,endpoint=False):
    assert isinstance(data,np.ndarray) and len(data.shape) == 1, "Vector packet must be 1-d NumPy array."
    super().__init__(data,endpoint) 

  @property
  def dtype(self):
    return str(self.data.dtype)

## BVector packet hold 1-d array data with bytes format.
class BVector(Packet):

  def __init__(self,data,dtype,endpoint=False):
    assert isinstance(data,bytes), "BVector packet must be bytes object."
    assert dtype in ["int16","float32"]
    super().__init__(data,endpoint)
    self.__dtype = dtype
  
  @property
  def dtype(self):
    return self.__dtype
  
  def decode(self):
    return Vector(np.frombuffer(self.data,dtype=self.__dtype),endpoint=self.is_endpoint())

## Text packet hold the top 1 best decoding result.
class Text(Packet):

  def __init__(self,data,endpoint=False):
    assert isinstance(data,str), "Text packet must be string."
    super().__init__(data,endpoint)
  
  @property
  def dtype(self):
    return "str"

## Component is used to process packets
class Component(StateFlag):

  def __init__(self,name="compnent"):
    super().__init__()
    self.__name = name
    self.__coreThread = None

  @property
  def coreThread(self)->threading.Thread:
    return self.__coreThread

  @property
  def name(self):
    return self.__name
  
  # Please implement it
  @property
  def outPIPE(self)->PIPE:
    raise Exception("Please implement this function.")
  
  def start(self,inPIPE:PIPE):
    '''Start run a thread to process packets in inPIPE.'''
    self.shift_state_to_alive()
    self.__coreThread = self._start(inPIPE)
    assert isinstance(self.__coreThread,threading.Thread), "The function _start must return a threading.Thread object!"
  
  # Please implement it
  # This function must return a threading.Thread object.
  def _start(self,inPIPE)->threading.Thread:
    raise Exception("Please implement this function.")

  def stop(self):
    '''Terminate this component normally.'''
    self.shift_state_to_termination()
    self.outPIPE.stop()
  
  def kill(self):
    '''Terminate this component with error state.'''
    self.shift_state_to_error()
    self.outPIPE.kill()
  
  def wait(self):
    '''Wait until thread finished.'''
    if self.coreThread is None:
      raise Exception("Component has not been started.")
    else:
      self.coreThread.join()

## Chain is a container to manage the sequential Component-PIPEs
class Chain(StateFlag):

  def __init__(self):
    super().__init__()
    self.__chain = []
    self.__name2id = {}

  def check_chain(self):
    '''Do some checks'''
    assert len(self.__chain) > 0, "Chain is empty."

  def add(self,component):
    '''Add a new component to the tail of chain.'''
    assert self.is_silent(), "Can only add new component into a silent chain."
    assert isinstance(component,Component), "Need Component object."
    self.__chain.append(component)
  
  def start(self):
    # check if this is a empty chain.
    self.check_chain()
    # link and run all components
    previousPIPE = None
    for i in range(len(self.__chain)):
      self.__chain[i].start(inPIPE=previousPIPE)
      previousPIPE = self.__chain[i].outPIPE
    # set silent flag
    self.shift_state_to_alive()
  
  def stop(self):
    # check if this is a empty chain.
    self.check_chain()
    # stop the first component and wait the last component
    self.__chain[0].stop()
    self.__chain[-1].wait()
    # set chain state
    self.shift_state_to_termination()
  
  def kill(self):
    # check if this is a empty chain.
    self.check_chain()
    # kill all components
    for i in range(len(self.__chain)):
      self.__chain[i].kill()
    # set chain state
    self.shift_state_to_error()

  def wait(self):
    # check if this is a empty chain.
    self.check_chain()
    # wait the last component
    self.__chain[-1].wait()
  
  @property
  def outPIPE(self):
    # check the chain
    self.check_chain()
    # return the output PIPE
    return self.__chain[-1].outPIPE
  
  def get_component(self,name)->Component:
    if name not in self.__name2id.keys():
      raise Exception(f"No such component: {name}")
    ID = self.__name2id[name]
    return self.__chain[ID]

  def get(self):
    # check the chain
    self.check_chain()
    # Get a packet from output PIPE
    return self.__chain[-1].outPIPE.get()

# A tool for debug
def wait_and_dynamic_display(target,items=["is_endpoint","data"]):
  '''
  Wait the target and display the packets of its outPIPE dynamically.

  Args:
    _target_: a Component or Chain object.
    _items_: choose what info to display. All items must be name of attributes or arguments-free methods.  
            Or it can be a dict of functions to process the packet, like
            wait_and_dynamic_display(target,items={"is_endpoint":lambda pac,name:pac.is_endpoint()},)
  '''
  assert isinstance(target,(Component,Chain)),"<target> should be a Component or Chain object."
  assert target.is_alive(), "<target> is not alive."
  assert isinstance(items,(list,dict)),"<items> should be a list of names or dict of functions."

  def default_function(pac,name):
    tar = getattr(pac,name)
    return tar() if callable(tar) else tar

  if isinstance(items,list):
    items = dict( (name,lambda pac,na:default_function(pac,na)) for name in items )

  while True:
    if target.outPIPE.is_error() or target.outPIPE.is_exhaustion():
      break
    elif target.outPIPE.is_empty():
      time.sleep(info.TIMESCALE)
    else:
      packet = target.outPIPE.get()

      for name in items.keys():
        print(f"{name}: ", items[name](packet,name) )
      print()

## Define how to encode the vector data in order to send to subprocess
def encode_vector(vec)->bytes:
  return (" " + " ".join( map(str,vec)) + " ").encode()

## A simple function to run shell command
def run_exkaldi_shell_command(cmd,inputs=None):
  '''
  _inputs_: None or bytes object.
  '''
  if inputs is not None:
    assert isinstance(inputs,bytes),""
    stdin = subprocess.PIPE
  else:
    stdin = None

  p = subprocess.Popen(cmd,shell=True,stdin=stdin,stderr=subprocess.PIPE,stdout=subprocess.PIPE)
  (out,err) = p.communicate(input=inputs)
  cod = p.returncode

  if cod != 0:
    raise Exception(err.decode())
  else:
    out =  out.decode().strip().split()
    return out
