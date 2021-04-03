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
import ctypes
import time
import datetime
from collections import namedtuple

from exkaldirt.version import version

class Info:
  '''
  A object to define some parameters of ExKaldi-RT.
  '''
  def __init__(self):
    self.__kaldi_existed = False
    self.__timeout = 1800
    self.__timescale = 0.01
    self.__max_socket_buffer_size = 10000
    # Check Kaldi root directory and ExKaldi-RT tool directory
    self.__kaldi_root = self.__find_kaldi_root()
    self.__cmdroot = os.path.join(self.__kaldi_root,"src","exkaldirtcbin")
    # Get the float floor
    self.__epsilon = self.__get_floot_floor()

  @property
  def KALDI_EXISTED(self)->bool:
    return self.__kaldi_existed

  @property
  def VERSION(self):
    return version

  @property
  def CMDROOT(self):
    return self.__cmdroot

  @property
  def KALDI_ROOT(self):
    return self.__kaldi_root

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
    '''Maximum times to resend the packet if packet is lost'''
    return 10
  
  @property
  def MAX_SOCKET_BUFFER_SIZE(self):
    return self.__max_socket_buffer_size

  def set_MAX_SOCKET_BUFFER_SIZE(self,size:int):
    assert isinstance(size,int) and size > 4
    self.__max_socket_buffer_size = size

  def set_TIMEOUT(self,value):
    assert isinstance(value,int) and value > 0, "TIMEOUT must be an int value."
    self.__timeout = value
  
  def set_TIMESCALE(self,value):
    assert isinstance(value,float) and 0 < value < 1.0, "TIMESCALE should be a float value in (0,1)."
    self.__timescale = value

  def __find_kaldi_root(self):
    '''Look for the kaldi root path.'''
    if "KALDI_ROOT" in os.environ.keys():
      KALDI_ROOT = os.environ["KALDI_ROOT"]
      self.__kaldi_existed = True
    else:
      cmd = "which copy-matrix"
      p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      out, err = p.communicate()
      if out == b'':
        print( "Warning: Kaldi root directory was not found automatically. " + \
               "Module: exkaldirt.feature and exkaldirt.decode are unavaliable."
              )
        #raise Exception("Kaldi root directory was not found automatically. Please ensure it has been added in environment sucessfully.")
      else:
        out = out.decode().strip()
        KALDI_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(out)))
        self.__kaldi_existed = True

    if self.__kaldi_existed:
      assert os.path.isfile(os.path.join(KALDI_ROOT,"src","exkaldirtcbin","exkaldi-online-decoder")), \
            "ExKaldi-RT C++ source files have not been compiled. " + \
            "Please consult the installation in github: https://github.com/wangyu09/exkaldi-rt ."

      return KALDI_ROOT 
    else:
      return None

  def __get_floot_floor(self):
    '''Get the floot floor value'''
    if self.__cmdroot is None:
      return 1.19209e-07
    else:
      cmd = os.path.join(self.__cmdroot,"get-float-floor")
      p = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
      (out,err) = p.communicate()
      out = out.decode().strip()
      if len(out) == 0:
        raise Exception("Failed to get float floor:\n" + err.decode())
      return float(out)

# Instantiate this object.
info = Info()

class KillableThread(threading.Thread):
  '''
  This is a killable thread object
  '''
  def get_id(self)->int:
    '''Return the thread id.'''
    if not self.isAlive():
      raise threading.ThreadError("The thread is not active.")

    if hasattr(self,'_thread_id'):
      return self._thread_id
    for ID, thread in threading._active.items():
      if thread is self:
        return ID

    raise AssertionError("Could not determine the thread's id.")

  def kill(self):
    '''Kill this thread forcely.'''
    tid = ctypes.c_long( self.get_id() )
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid,ctypes.py_object(SystemExit)) 
    if res == 0:
      raise ValueError("Invalid thread id.")
    elif res > 1: 
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None) 
        raise SystemError("PyThreadState_SetAsyncExc failed.")

class ExKaldiRTBase:
  '''
  Base class to describe state of Components and PIPEs.
  '''
  # State Flags
  SILENT = 0
  ALIVE = 1
  TERMINATED = 2
  WRONG = 3
  # Object Counter
  OBJ_COUNTER = 0

  def __init__(self,name=None):
    # State flag
    self.__state = ExKaldiRTBase.SILENT
    # Name it
    self.rename(name=name)

  @property
  def name(self):
    return self.__name

  def rename(self,name=None):
    '''
    Rename it.
    '''
    if name is None:
      name = self.__class__.__name__
    else:
      assert isinstance(name,str) and len(name) > 0, f"<name> must be a string but got: {name}."
    self.__name = name + f"[{ExKaldiRTBase.OBJ_COUNTER}]"
    ExKaldiRTBase.OBJ_COUNTER += 1

  def is_silent(self) -> bool:
    return self.__state == ExKaldiRTBase.SILENT

  def is_alive(self) -> bool:
    return self.__state == ExKaldiRTBase.ALIVE
  
  def is_wrong(self) -> bool:
    return self.__state == ExKaldiRTBase.WRONG
  
  def is_terminated(self) -> bool:
    return self.__state == ExKaldiRTBase.TERMINATED
  
  def shift_state_to_silent(self):
    self.__state = ExKaldiRTBase.SILENT

  def shift_state_to_alive(self):
    self.__state = ExKaldiRTBase.ALIVE

  def shift_state_to_wrong(self):
    self.__state = ExKaldiRTBase.WRONG

  def shift_state_to_terminated(self):
    self.__state = ExKaldiRTBase.TERMINATED

########################################

class Packet:
  '''
  Packet object is used to hold various stream data, such as audio stream, feature and probability.
  These data will be processed by Component and passed in PIPE.
  '''
  def __init__(self,data):
    self.__data = data

  @property
  def data(self):
    return self.__data

  @property
  def dtype(self):
    raise Exception("Please implement this function.")

class Element(Packet):
  '''
  Element packet hold single value data.
  '''
  def __init__(self,data):
    if isinstance(data,int):
      data = np.int16(data)
    elif isinstance(data,float):
      data = np.int32(data)
    else:
      assert isinstance(data,(np.int8,np.int16,np.int32,
                             np.float16,np.float32,np.float64)), "Element packet must be int or float value."
    super().__init__(data)
  
  @property
  def dtype(self) -> str:
    return str(self.data.dtype)

class Vector(Packet):
  '''
  Vector packet hold 1-d NumPy array data.
  '''
  def __init__(self,data):
    assert isinstance(data,np.ndarray) and len(data.shape) == 1, "Vector packet must be 1-d NumPy array."
    super().__init__(data) 

  @property
  def dtype(self) -> str:
    return str(self.data.dtype)

class Text(Packet):
  '''
  Text packet hold the top 1-best decoding result.
  '''
  def __init__(self,data):
    assert isinstance(data,str), "Text packet must be string."
    super().__init__(data)
  
  @property
  def dtype(self) -> str:
    return "str"

class Endpoint(Packet):
  '''
  A special flag to mark endpoint.
  '''
  pass

# Instantiate this object.
ENDPOINT = Endpoint(None)

def is_endpoint(obj):
  '''
  If this is Endpoint, return True.
  '''
  return isinstance(obj,Endpoint)

########################################

class PIPE(ExKaldiRTBase):
  '''
  PIPE is used to connect Components and pass Packets.
  It is a Last-In-Last-Out queue.
  Note that we will forcely:
  1. remove continuous Endpoint flags.
  2. discard the head packet if it is Endpoint flag.
  '''
  def __init__(self,name=None):
    # Initial state and name
    super().__init__(name=name)
    # Set cache
    self.__cache = queue.Queue()
    # Set some flags
    self.reset()

  def reset(self):
    '''
    Initialize or reset some flags.
    '''
    self.__cache.queue.clear()
    self.__extra_info = None
    self.shift_state_to_alive()
    # a flag to remove continue ENDPOINT or head ENDPOINT 
    self.__last_added_endpoint = True 
    # a flag to mark whether this PIPE is blocked
    self.__blocked = False 
    # flags to report time points
    self.__firstPut = None
    self.__lastPut = None
    self.__firstGet = None
    self.__lastGet = None
  
  def kill(self):
    '''
    Kill this PIPE with state: WRONG.
    '''
    self.shift_state_to_wrong()
    self.__cache.queue.clear()
    self.__extra_info = None
  
  def stop(self):
    '''
    Stop this PIPE state with: TERMINATED.
    You can still get data from this PIPE till it becomes exhaustion.
    '''
    self.shift_state_to_terminated()
    # unblock the PIPE automatically
    self.unblock() 
  
  def is_empty(self)->bool:
    '''
    If there is no any data in PIPE, return True.
    '''
    return self.__cache.empty()

  def is_blocked(self)->bool:
    '''
    Return True if PIPE is blocked.
    '''
    return self.__blocked

  def is_exhausted(self)->bool:
    '''
    If there is no more data in PIPE, return True.
    '''
    return self.is_terminated() and self.is_empty()

  def clear(self):
    '''
    Clear the cache.
    '''
    self.__cache.queue.clear()
  
  def size(self):
    '''
    Get the size.
    '''
    return self.__cache.qsize()
  
  def get(self)->Packet:
    '''
    Pop a packet from head.
    '''
    if self.is_exhausted():
      raise Exception(f"{self.name}: No more data in PIPE.") 
    elif self.is_wrong():
      raise Exception(f"{self.name}: Can not get packet from a wrong PIPE." )
    elif self.is_blocked():
      raise Exception(f"{self.name}: Can not get packet from a blocked PIPE.") 
    
    packet = self.__cache.get(timeout=info.TIMEOUT)

    if self.__firstGet is None:
      self.__firstGet = datetime.datetime.now()
    self.__lastGet = datetime.datetime.now()

    return packet
  
  def put(self,packet):
    '''
    Push a new packet to tail.
    Note that: we will remove the continuous Endpoint.
    '''
    if not self.is_alive():
      raise Exception(f"{self.name}: Can only put packet into an alive PIPE." )
    assert isinstance(packet,Packet), f"{self.name}: Try to put a not-Packet object into PIPE."
    
    # record time stamp
    if self.__firstPut is None:
      self.__firstPut = datetime.datetime.now()
    self.__lastPut = datetime.datetime.now()
      
    if is_endpoint(packet):
      if not self.__last_added_endpoint:
        self.__cache.put(packet)
        self.__last_added_endpoint = True
    else:
      self.__cache.put(packet)
      self.__last_added_endpoint = False
  
  def add_extra_info(self,info=None):
    '''
    Add any extra information to PIPE.
    '''
    self.__extra_info = info
  
  def get_extra_info(self):
    '''
    Get the extra information storaged in PIPE.
    '''
    return self.__extra_info

  def to_list(self)->list:
    '''
    Convert PIPE to lists divided by Endpoint.
    Only terminated PIPE can be converted.
    '''
    assert self.is_terminated(), f"{self.name}: Only terminated PIPE can be converted to list."

    size = self.size()
    result = []
    partial = []
    for i in range(size):
      packet = self.__cache.get()
      if is_endpoint(packet) and len(partial)>0:
        result.append( partial )
        partial = []
      else:
        partial.append( packet.data )
    if len(partial)>0:
      result.append( partial )
    return result[0] if len(result) == 1 else result

  def block(self):
    '''Block this PIPE so that it is unable to get data from it until unblocked or terminated.'''
    self.__blocked = True
  
  def unblock(self):
    '''
    Unblock PIPE.
    '''
    self.__blocked = False

  def report_time(self):
    '''
    Report time information.
    '''
    return namedtuple("TimeReport",["name","firstPut","lastPut","firstGet","lastGet"])(
                  self.name,self.__firstPut,self.__lastPut,self.__firstGet,self.__lastGet
                )

class Component(ExKaldiRTBase):
  '''
  Components are used to process Packets.
  '''
  def __init__(self,name=None):
    # Initial state and name
    super().__init__(name=name)
    # Define an output PIPE
    self.__outPIPE = PIPE(name=self.name+" output PIPE")
    # Each Component has a core thread to run a function to process packets.
    self.__coreThread = None

  def reset(self):
    '''
    Clear and reset Component.
    '''
    if self.is_alive():
      raise Exception(f"{self.name}: Can not reset a ALIVE Component, please stop it firstly.")
    self.__coreThread = None
    self.outPIPE.reset()
    self.shift_state_to_silent()

  @property
  def coreThread(self)->KillableThread:
    '''
    Get the core thread.
    '''
    return self.__coreThread

  @property
  def outPIPE(self)->PIPE:
    '''Get the output PIPE.'''
    return self.__outPIPE
  
  def start(self,inPIPE:PIPE):
    '''
    Start running a thread to process Packets in inPIPE.
    '''
    self.shift_state_to_alive()
    self.__coreThread = self._start(inPIPE=inPIPE)
    assert isinstance(self.__coreThread,threading.Thread), f"{self.name}: The function _start must return a threading.Thread object!"
  
  # Please implement it
  # This function must return a threading.Thread object.
  def _start(self,inPIPE)->threading.Thread:
    raise Exception(f"{self.name}: Please implement the ._start function.")

  def stop(self):
    '''
    Terminate this component normally.
    Note that we do not terminate the core thread by this function.
    We hope the core thread can be terminated with a mild way.
    '''
    # Change state
    self.shift_state_to_terminated()
    # Append an ENDPOINT flag into output PIPE then terminate it
    self.outPIPE.put( ENDPOINT )
    self.outPIPE.stop()
  
  def kill(self):
    '''
    Terminate this component with state: WRONG.
    It means errors occurred somewhere.
    Note that we do not kill the core thread by this function.
    We hope the core thread can be terminated with a mild way.
    '''
    # Change state
    self.shift_state_to_wrong()
    # Kill output PIPE in order to pass error state to the succeeding components
    self.outPIPE.kill()
  
  def wait(self):
    '''
    Wait until the core thread is finished.
    '''
    if self.coreThread is None:
      raise Exception(f"{self.name}: Component has not been started.")
    else:
      self.coreThread.join()

  def block(self):
    '''Block the output PIPE.'''
    self.outPIPE.block()
  
  def is_blocked(self):
    '''Return True if the output PIPE is blocked.'''
    self.outPIPE.is_blocked()
  
  def unblock(self):
    '''Unblock the output PIPE.'''
    self.outPIPE.unblock()

class Chain(ExKaldiRTBase):
  '''
  Chain is a container to easily manage the sequential Component-PIPEs.
  '''
  def __init__(self,name=None):
    # Initial state and name
    super().__init__(name=name)
    # A container to hold components
    self.__chain = []
    # Mark the component
    self.__blockedFlag = []
    # Component name -> Index
    self.__name2id = {}

  def reset(self):
    '''
    Reset the Chain.
    We will reset all components in it.
    '''
    if self.is_alive():
      raise Exception(f"{self.name}: Can not reset a ALIVE Chain, please stop it firstly.")
    # Reset all components
    for comp,block in zip(self.__chain,self.__blockedFlag):
      comp.reset()
      if block:
        comp.block()
    # Reset State
    self.shift_state_to_silent()

  def check_chain(self):
    '''
    Do some checks.
    '''
    assert len(self.__chain) > 0, f"{self.name}: Chain is empty."

  def add(self,component,block=False):
    '''
    Add a new component to the tail of chain.
    
    Args:
      _block_: If True, we will block this Component.
    '''
    assert self.is_silent(), f"{self.name}: Can only add new component into a silent chain."
    assert isinstance(component,Component), f"{self.name}: <component> of .add method must be a Component object."
    assert isinstance(block,bool), f"{self.name}: <block>  of .add method be a bool value."
    if block:
      component.block()
    self.__chain.append(component)
    self.__blockedFlag.append(block)
  
  def start(self,inPIPE=None):
    '''
    Start processing thread.
    '''
    self.check_chain()
    # Link and run all components.
    previousPIPE = inPIPE
    for i in range(len(self.__chain)):
      self.__chain[i].start(inPIPE=previousPIPE)
      previousPIPE = self.__chain[i].outPIPE
    # Set state
    self.shift_state_to_alive()
  
  def stop(self):
    '''
    Stop processing thread normally.
    '''
    # Check.
    self.check_chain()
    # Stop the first Component.
    self.__chain[0].stop()
    #self.__chain[-1].wait()
    # Set chain state.
    self.shift_state_to_terminated()
  
  def kill(self):
    '''
    Kill processing thread with error state.
    '''
    # check.
    self.check_chain()
    # Kill all components.
    for i in range(len(self.__chain)):
      self.__chain[i].kill()
    # Set chain state.
    self.shift_state_to_wrong()

  def wait(self):
    '''
    Wait processing thread.
    '''
    # Check.
    self.check_chain()
    # Wait the last Component.
    self.__chain[-1].wait()
    # Change state
    self.shift_state_to_terminated()
  
  @property
  def outPIPE(self)->PIPE:
    '''
    Get the output PIPE.
    '''
    # Check.
    self.check_chain()
    # Return the output PIPE.
    return self.__chain[-1].outPIPE
  
  def component(self,name=None,ID=None)->Component:
    '''
    Get the component by calling its name.
    
    Args:
      _name_: the name of Component.
      _ID_: the index number of Component.
    '''
    assert not (name is None and ID is None), f"{self.name}: Both <name> and <ID> are None."

    if name is not None:
      assert ID is None
      if name not in self.__name2id.keys():
        raise Exception(f"{self.name}: No such Component: {name}")
      ID = self.__name2id[name]
      return self.__chain[ID]
    else:
      assert isinstance(ID,int)
      return self.__chain[ID]

  def get(self)->Packet:
    '''
    Get a Packet from the output PIPE.
    '''
    self.check_chain()
    return self.__chain[-1].outPIPE.get()

  # Overwrite this function
  def is_wrong(self):
    if super().is_wrong():
      return True
    else:
      for comp in self.__chain:
        if comp.is_wrong():
          return True
      
      return False
  
  # Overwrite this function
  def is_terminated(self):
    return super().is_terminated() or self.__chain[-1].is_terminated()

def dynamic_run(target,inPIPE=None,items=["data"]):
  '''
  This is a tool for debug or testing.
  Wait the target and display the packets of its outPIPE dynamically.

  Args:
    _target_: a Component or Chain object.
    _inPIPE_: the input PIPE if necessary.
    _items_: choose what info to display. All items must be name of attributes or arguments-free methods.  
            Or it can be a dict of functions to process the Packet, like:
            wait_and_dynamic_display(target,items={"data-shape":lambda x:x.data.shape})
  '''
  assert isinstance(target,(Component,Chain)),"<target> should be a Component or Chain object."
  assert isinstance(items,(list,dict)),"<items> should be a list of names or dict of functions."
  assert inPIPE is None or isinstance(inPIPE,PIPE), "<inPIPE> should be None or a PIPE object."

  if target.is_silent():
    target.start(inPIPE=inPIPE)
  else:
    assert target.is_alive(), "<target> must be SILENT or ALIVE Component or Chain."

  def default_function(pac,name):
    tar = getattr(pac,name)
    return tar() if callable(tar) else tar

  while True:
    if target.outPIPE.is_wrong() or target.outPIPE.is_exhausted():
      break
    elif target.outPIPE.is_empty():
      time.sleep(info.TIMESCALE)
    else:
      packet = target.outPIPE.get()
      if is_endpoint(packet):
        print(f"----- Endpoint -----")
        continue
      if isinstance(items,list):
        for name in items:
          print(f"{name}: ", default_function(packet,name) )
      else:
        for name in items.keys():
          print(f"{name}: ", items[name](packet) )
      print()
  
  target.wait()

  #print("########## Time Report ##########")
  #if inPIPE is not None:
  #  st = inPIPE.report_time().firstGet
  #elif isinstance(target,Chain):
  #  st = target.component(ID=0).outPIPE.report_time().firstPut
  #else:
  #  st = target.outPIPE.report_time().firstPut
  #et = target.outPIPE.report_time().lastPut
  #print(f"Start Time: {st.year}-{st.month}-{st.day}, {st.hour}:{st.minute}:{st.second}:{st.microsecond}")
  #print(f"End Time: {et.year}-{et.month}-{et.day}, {et.hour}:{et.minute}:{et.second}:{et.microsecond}")
  #print(f"Time Cost: {(et-st).total_seconds()} seconds")
  #print("#################################")

def encode_vector(vec)->bytes:
  '''
  Define how to encode the vector data in order to send to subprocess.
  '''
  return (" " + " ".join( map(str,vec)) + " ").encode()

def run_exkaldi_shell_command(cmd,inputs=None)->list:
  '''
  A simple function to run shell command.

  Args:
    _cmd_: a string of a shell command and its arguments.
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
