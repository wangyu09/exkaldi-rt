# coding=utf-8
#
# Yu Wang (University of Yamanashi)
# Apr, 2021
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
import multiprocessing
import ctypes
import time
import random
import datetime
from collections import namedtuple

#from exkaldirt.version import version
#from exkaldirt.utils import *

from version import version
from utils import *

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
    self.__cmdroot = None
    if self.__kaldi_root is not None:
      self.__cmdroot = os.path.join(self.__kaldi_root,"src","exkaldirtcbin")
    # Get the float floor
    self.__epsilon = self.__get_floot_floor()

  def __find_kaldi_root(self):
    '''Look for the ExKaldi-RT C++ command root path.'''
    if "KALDI_ROOT" in os.environ.keys():
      KALDI_ROOT = os.environ["KALDI_ROOT"]
      self.__kaldi_existed = True
    else:
      cmd = "which copy-matrix"
      p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      out, err = p.communicate()
      if out == b'':
        print( "Warning: Kaldi root directory was not found automatically. " + \
               "Module, exkaldirt.feature and exkaldirt.decode, are unavaliable." 
              )
      else:
        out = out.decode().strip()
        # out = "/yourhome/kaldi/src/bin/copy-matrix"
        KALDI_ROOT = os.path.dirname( os.path.dirname( os.path.dirname(out)) )
        self.__kaldi_existed = True

    if self.__kaldi_existed:
      cmdroot = os.path.join(KALDI_ROOT,"src","exkaldirtc")
      assert os.path.isfile(os.path.join(cmdroot,"exkaldi-online-decoder")), \
            "ExKaldi-RT C++ source files have not been compiled sucessfully. " + \
            "Please consult the Installation in github: https://github.com/wangyu09/exkaldi-rt ."
      
      return KALDI_ROOT

    else:
      return None

  def __get_floot_floor(self):
    '''Get the floot floor value.'''
    if self.__cmdroot is None:
      return 1.19209e-07
    else:
      sys.path.append( self.__cmdroot )
      import cutils
      return cutils.get_float_floor()

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

# Instantiate this object.
info = Info()

class ExKaldiRTBase:
  '''
  Base class of ExKaldi-RT.
  '''
  # Object Counter
  OBJ_COUNTER = 0

  def __init__(self,name=None):
    # Give an unique ID for this object.
    self.__objid = ExKaldiRTBase.OBJ_COUNTER
    ExKaldiRTBase.OBJ_COUNTER += 1
    # Name it
    self.__naming(name=name)

  def __naming(self,name=None):
    '''
    Name it.
    '''
    if name is None:
      name = self.__class__.__name__
    else:
      assert isinstance(name,str) and len(name) > 0, f"_name_ must be a string but got: {name}."
      #assert " " not in name, "_name_ can not include space."
      
    self.__name = name + f"[{self.__objid}]"

  @property
  def objid(self):
    return self.__objid

  @property
  def name(self):
    return self.__name

########################################

class Packet:
  '''
  Packet object is used to hold various stream data, such as audio stream, feature and probability.
  These data will be processed by Component and passed in PIPE.
  We only support 4 types of data: int, float, str, np.ndarray.
  '''
  def __init__(self,items,sid,eid,idmaker,mainKey=None):
    assert isinstance(items,dict), f"_items_ must be a dict object."
    self.__data = {}
    # Set items
    if mainKey is not None:
      assert mainKey in items.keys()
      self.__mainKey = mainKey
      for key,value in items.items():
        self.add(key,value)
    else:
      self.__mainKey = None
      for key,value in items.items():
        self.add(key,value)
        if self.__mainKey is None:
          self.__mainKey = key
    # Set start id and end id
    self.update_id(sid,eid)
    assert isinstance(idmaker,int)
    self.__idmaker = idmaker

  @property
  def mainKey(self):
    return self.__mainKey
      
  @property
  def sid(self)->int:
    return self.__sid
  
  @property
  def eid(self)->int:
    return self.__eid 
  
  @property
  def idmaker(self)->str:
    return self.__idmaker
  
  def update_id(self,sid,eid):
    assert isinstance(sid,int) and sid >= 0
    self.__sid = sid
    assert isinstance(eid,int) and eid >= sid
    self.__eid = eid
  
  def __getitem__(self,key=None):
    assert key in self.__data.keys()
    return self.__data[key]
  
  def add(self,key,data,asMainKey=False):
    # Verify key name
    assert isinstance(key,str), "_key_ must be a string."
    assert " " not in key, "_key_ can not include space."
    # Verify value
    if isinstance(data,int):
        data = np.int16(data)
    elif isinstance(data,float):
        data = np.float32(data)
    elif isinstance(data,(np.signedinteger,np.floating)):
        pass
    elif isinstance(data,np.ndarray):
        assert len(data.shape) == 1
        data = data.copy()
    else:
        raise Exception("Unsupported data type")
    self.__data[key] = data
    if asMainKey:
      self.__mainKey = key
  
  def encode(self)->bytes:
    '''
    Encode packet.
    '''
    result = b""
    
    # Encode idmaker
    result += uint_to_bytes(self.idmaker, length=4)
    # Encode sid and eid
    result += uint_to_bytes(self.sid, length=4)
    result += uint_to_bytes(self.eid, length=4)
    # Encode main key
    result += (self.mainKey.encode() + b" ")
    
    # Encode data
    for key,value in self.__data.items():
      # Encode key
      result += key.encode() + b" "
      if isinstance( value,(np.signedinteger,np.floating) ):
        bvalue = element_to_bytes( value )
        bvalue = b"E" + uint_to_bytes( len(bvalue),length=1 ) + bvalue
      elif isinstance(value,np.ndarray):
        bvalue = vector_to_bytes( value )
        bvalue = b"V" + uint_to_bytes( len(bvalue),length=4 ) + bvalue
      elif isinstance(value,str):
        bvalue = value.encode()
        bvalue = b"S" + uint_to_bytes( len(bvalue),length=4 ) + bvalue
      else:
        raise Exception("Unsupported data type.")
      result += bvalue
  
    return result

  @classmethod
  def decode(cls,bstr):
    '''
    Generate a packet object.
    '''
    with BytesIO(bstr) as sp:
      # Read ID maker
      maker = uint_from_bytes( sp.read(4) )
      # Read ID
      sid = uint_from_bytes( sp.read(4) )
      eid = uint_from_bytes( sp.read(4) )
      # Read main key
      mainKey = read_string( sp )
      # Read data
      result = {}
      while True:
        key = read_string(sp)
        if key == "":
          break
        flag = sp.read(1).decode()
        if flag == "E":
          size = uint_from_bytes( sp.read(1) )
          data = element_from_bytes( sp.read(size) )
        elif flag == "V":
          size = uint_from_bytes( sp.read(4) )
          data = vector_from_bytes( sp.read(size) )
        elif flag == "S":
          size = uint_from_bytes( sp.read(4) )
          data = sp.read(size).decode()
        else:
          raise Exception(f"Unknown flag: {flag}")

        result[ key ] = data
  
    return Packet(result,sid=sid,eid=eid,idmaker=maker,mainKey=mainKey)
  
  def keys(self):
    return self.__data.keys()
  
  def values(self):
    return self.__data.values()
  
  def items(self):
    return self.__data.items()

# ENDPOINT is a special packet.
class Endpoint(Packet):
  def __init__(self):
    super().__init__({},0,0,-1)

ENDPOINT = Endpoint()

def is_endpoint(obj):
  '''
  If this is Endpoint, return True.
  '''
  return isinstance(obj,Endpoint)

# Standerd output lock
stdout_lock = multiprocessing.Lock()

def print_(*args,**kwargs):
  with stdout_lock:
    print(*args,**kwargs)

########################################
mark = namedtuple("Mark",["silent","active","terminated","wrong","stranded","endpoint","inPIPE","outPIPE"])(
                            0,1,2,3,4,5,6,7,)

# silent : PIPE is unavaliable untill it is activated.
# active | stranded : There might be new packets appended in it later. 
# wrong  | terminated : Can not add new packets in PIPE but can still get packets from it.

class PIPE(ExKaldiRTBase):
  '''
  PIPE is used to connect Components and pass Packets.
  It is a Last-In-Last-Out queue.
  It is designed to exchange data and state between mutiple processes.
  Note that we will forcely:
  1. remove continuous Endpoint flags.
  2. discard the head packet if it is Endpoint flag.
  '''
  def __init__(self,name=None):
    # Initilize state and name
    super().__init__(name=name)
    # Set a cache to pass data
    self.__cache = multiprocessing.Queue()
    # Flags used to communicate between diffrent components (with mutiple processes)
    self.__state = cState(mark.silent)
    self.__inlocked = cBool(False)
    self.__outlocked = cBool(False)
    self.__last_added_endpoint = cBool(False)
    self.__firstPut = cDouble(0.0)
    self.__lastPut = cDouble(0.0)
    self.__firstGet = cDouble(0.0)
    self.__lastGet = cDouble(0.0)
    self.__time_stamp = cDouble( time.time() )
    # Password to access this PIPE
    self.__password = random.randint(0,100)
    # Class backs functions
    self.__callbacks = []

  def state_is_(self,*m) -> bool:
    return self.__state.value in m

  def __shift_state_to_(self,m):
    assert m in mark
    self.__state.value = m
    self.__time_stamp.value = time.time()

  @property
  def state(self):
    return self.__state.value

  @property
  def timestamp(self):
    return self.__time_stamp.value

  #############
  # Lock input or output port
  #############

  def is_inlocked(self)->bool:
    return self.__inlocked.value
  
  def is_outlocked(self)->bool:
    return self.__outlocked.value

  def lock_in(self)->int:
    '''
    Lock this input of PIPE.
    '''
    if self.is_inlocked():
      return None
    self.__inlocked.value = True
    return self.__password
  
  def lock_out(self)->int:
    '''
    Lock this output of PIPE.
    '''
    if self.is_outlocked():
      return None
    self.__outlocked.value = True
    return self.__password

  def release_in(self,password):
    if self.is_inlocked:
      if password == self.__password:
        self.__inlocked.value = False
      else:
        print_(f"{self.name}: Wrong password to release input port!")
  
  def release_out(self,password):
    if self.is_outlocked:
      if password == self.__password:
        self.__outlocked.value = False
      else:
        print_(f"{self.name}: Wrong password to release output port!")

  #############
  # Some operations
  #############

  def clear(self):
    assert not self.state_is_(mark.active), f"{self.name}: Can not clear a active PIPE."
    # Clear
    size = self.__cache.qsize()
    for i in range(size):
      self.__cache.get()
  
  def reset(self):
    '''
    Do:
      1. clear data,
      2. reset state to silent,
      3. reset endpoint and time information.
    Do not:
      1. reset input lock and output lock flags.
      2. reset the callbacks. 
    '''
    assert not (self.state_is_(mark.active) or self.state_is_(mark.stranded)), \
          f"{self.name}: Can not reset a active or stranded PIPE."
    # Clear cache
    self.clear()
    # Reset state
    self.__shift_state_to_(mark.silent)
    # A flag to remove continue ENDPOINT or head ENDPOINT 
    self.__last_added_endpoint.value = False
    # flags to report time points
    self.__firstPut.value = 0.0
    self.__lastPut.value = 0.0
    self.__firstGet.value = 0.0
    self.__lastGet.value = 0.0

  def activate(self):
    '''
    State:  silent -> active
    '''
    assert self.state_is_(mark.silent)
    self.__shift_state_to_(mark.active)

  def kill(self):
    '''
    Kill this PIPE with state: wrong.
    '''
    assert self.state_is_(mark.active) or self.state_is_(mark.stranded)
    self.__shift_state_to_(mark.wrong)
  
  def stop(self):
    '''
    Stop this PIPE state with: terminated.
    '''
    assert self.state_is_(mark.active) or self.state_is_(mark.stranded)
    # Append a endpoint flag
    self.put( ENDPOINT, password=self.__password )
    # Shift state
    self.__shift_state_to_(mark.terminated)
  
  def pause(self):
    assert self.state_is_(mark.active), f"{self.name}: Can only pause active PIPE."
    self.__shift_state_to_(mark.stranded)

  def restart(self):
    assert self.state_is_(mark.stranded), f"{self.name}: Can only restart stranded PIPE."
    self.__shift_state_to_(mark.active)

  def size(self):
    '''
    Get the size.
    '''
    return self.__cache.qsize()

  def is_empty(self)->bool:
    '''
    If there is no any data in PIPE, return True.
    '''
    return self.__cache.empty()

  def is_exhausted(self)->bool:
    '''
    If there is no more data in PIPE, return True.
    '''
    return self.state_is_(mark.terminated,mark.wrong) and self.is_empty()

  def get(self,password=None,timeout=info.TIMEOUT)->Packet:
    '''
    Pop a packet from head.
    Can get packet from: active, wrong, terminated PIPE.
    Can not get packet from: silent and stranded PIPE. 
    '''
    if self.state_is_(mark.silent,mark.stranded):
      print_( f"Warning, {self.name}: Failed to get packet in PIPE. PIPE state is or silent or stranded." )
      return False

    assert not (self.state_is_(mark.silent) or self.state_is_(mark.stranded)), \
          f"{self.name}: Can not get packet from silent or stranded PIPE."
    # If PIPE is active and output port is locked
    if self.state_is_(mark.active) and self.is_outlocked():
      if password is None:
        raise Exception(f"{self.name}: Output of PIPE is clocked. Unlock or give the password to access it.")
      elif password != self.__password:
        raise Exception(f"{self.name}: Wrong password to access the PIPE.")
    
    packet = self.__cache.get(timeout=timeout)
    # Record time stamp
    if self.__firstGet.value == 0.0:
      self.__firstGet.value = time.time()
    self.__lastGet.value = time.time()
    # Return
    return packet
  
  def put(self,packet,password=None):
    '''
    Push a new packet to tail.
    Note that: we will remove the continuous Endpoint.
    Can put packet to: silent, alive.
    Can not put packet to: wrong, terminated and stranded PIPE.
    If this is a silent PIPE, activate it automatically.
    '''
    if self.state_is_(mark.wrong,mark.terminated,mark.stranded):
      print_( f"{self.name}: Failed to put packet in PIPE. PIPE state is not active or silent." )
      return False

    # If input port is locked
    if self.is_inlocked():
      if password is None:
        raise Exception(f"{self.name}: Input of PIPE is clocked. Unlock or give the password to access it.")
      elif password != self.__password:
        raise Exception(f"{self.name}: Wrong password to access the PIPE.")

    if self.state_is_(mark.silent):
      self.__shift_state_to_(mark.active)

    assert isinstance(packet,Packet), f"{self.name}: Only Packet can be appended in PIPE."
    
    # record time stamp
    if self.__firstPut.value == 0.0:
      self.__firstPut.value = time.time()
    self.__lastPut.value = time.time()
    # remove endpoint continuous flags and call back 
    if is_endpoint(packet):
      if not self.__last_added_endpoint.value:
        self.__cache.put(packet)
        self.__last_added_endpoint.value = True
    else:
      self.__cache.put(packet)
      self.__last_added_endpoint.value = False
      for func in self.__callbacks:
        func(packet)
    
    return True
  
  def to_list(self,mapFunc=None)->list:
    '''
    Convert PIPE to lists divided by Endpoint.
    Only terminated and wrong PIPE can be converted.
    '''
    assert self.state_is_(mark.terminated) or self.state_is_(mark.wrong), \
          f"{self.name}: Only terminated or wrong PIPE can be converted to list."
    # Check map function
    if mapFunc is None:
      mapFunc = lambda x:x[x.mainKey]
    else:
      assert callable(mapFunc)

    size = self.size()
    result = []
    partial = []
    for i in range(size):
      packet = self.__cache.get()
      if is_endpoint(packet) and len(partial) > 0:
        result.append( partial )
        partial = []
      else:
        partial.append( mapFunc(packet) )
    if len(partial)>0:
      result.append( partial )

    return result[0] if len(result) == 1 else result

  def report_time(self):
    '''
    Report time information.
    '''
    keys = ["name",]
    values = [self.name,]
    for name in ["firstPut","lastPut","firstGet","lastGet"]:
      value = getattr(self, f"_{type(self).__name__}__{name}").value
      if value != 0.0:
        keys.append(name)
        values.append(value)
    return namedtuple("TimeReport",keys)(*values)

  def callback(self,func):
    '''
    Add a callback function executing when a new packet is appended in PIPE.
    If _func_ is None, clear callback functions.
    '''
    assert self.state_is_(mark.silent)
    if func is None:
      self.__callbacks.clear()
    else:
      assert callable(func)
      self.__callbacks.append( func )

class NullPIPE(PIPE):

  def __init__(self,name=None):
    super().__init__(name=name)
    # Can not append and pop packets.
    self.lock_in()
    self.lock_out()

def is_nullpipe(pipe):
  '''
  If this is Endpoint, return True.
  '''
  return isinstance(pipe,NullPIPE)

class Tunnel(ExKaldiRTBase):

  def __init__(self,name=None):
    super().__init__(name=name)
    self.__tunnel = multiprocessing.Queue()
    # Password to access this PIPE
    self.__password = random.randint(0,100)
    self.__locked = cBool(False)
  
  def lock(self):
    if self.is_locked():
      return None
    else:
      self.__locked.value = True
      return self.__password
  
  def is_locked(self):
    return self.__locked.value

  def size(self):
    return self.__tunnel.qsize()

  def release(self,password):
    if self.is_locked():
      if password != self.__password:
        print_( f"{self.name}: Wrong password. Failed to release.")
      else:
        self.__locked.value = False

  def put(self,obj,password=None):
    # If input port is locked
    if self.is_locked():
      if password is None:
        raise Exception(f"{self.name}: Tunnel is clocked. Unlock or give the password to access it.")
      elif password != self.__password:
        raise Exception(f"{self.name}: Wrong password to access the Tunnel.")
    self.__tunnel.put(obj)

  def get(self,password=None):

    if self.is_locked():
      if password is None:
        raise Exception(f"{self.name}: Tunnel is clocked. Unlock or give the password to access it.")
      elif password != self.__password:
        raise Exception(f"{self.name}: Wrong password to access the Tunnel.")
    
    return self.__tunnel.get()

#def core_process_function(overFlag):
#  def _process_function(func):
#    def wrapper(*args,**kwargs):
#      func(*args, **kwargs)
#      overFlag.value = True
#    return wrapper
#  return _process_function

def main_process_function(func):
  def wrapper(*args,**kwargs):
    overFlag = kwargs.pop("overFlag")
    func(*args, **kwargs)
    overFlag.value = True
  return wrapper

class Component(ExKaldiRTBase):
  '''
  Components are used to process Packets.
  Components can only link to one input PIPE and has one output PIPE.
  '''
  def __init__(self,oKey=None,name=None):
    # Initial state and name
    super().__init__(name=name)
    # Define input and output PIPE
    # Input PIPE need to be linked
    self.__inPIPE = None
    self.__inPassword = None
    self.__outPIPE = PIPE(name=f"The output PIPE of "+self.name)
    self.__outPassword = self.__outPIPE.lock_in() # Lock the in-port of output PIPE
    # Each component has a core process to run a function to handle packets.
    self.__coreProcess = None
    # ID counter
    # Renumber the output frame if necessary
    self.__id_counter = cUint(0)
    # If need to redirect the input PIPE
    # We will stop the core process firstly and then link a new input PIPE and restart core process.
    self.__redirect_flag = cBool(False)
    # The input and output key name
    if oKey is not None:
      assert isinstance(oKey,str)
    self.iKey = None
    self.oKey = oKey
    # process over flag
    self.__core_process_over = cBool(False)

  @property
  def id_count(self):
    self.__id_counter.value += 1
    return self.__id_counter.value - 1

  def reset(self):
    '''
    Clear and reset Component.
    '''
    if self.coreProcess is None:
      return None
    elif self.coreProcess.is_alive():
      raise Exception(f"{self.name}: Component is active and can not reset. Please stop it firstly.")
    else:
      self.__coreProcess = None
      self.__outPIPE.reset()
      self.__id_counter.value = 0
      self.__core_process_over.value = False

  @property
  def coreProcess(self)->multiprocessing.Process:
    '''
    Get the core process.
    '''
    return self.__coreProcess

  @property
  def inPIPE(self)->PIPE:
    return self.__inPIPE

  @property
  def outPIPE(self)->PIPE:
    return self.__outPIPE
  
  def link(self,inPIPE:PIPE,iKey=None):
    assert isinstance(inPIPE,PIPE)
    # Release
    if self.__inPIPE is not None:
      assert not self.coreProcess.is_alive(), f"{self.name}: Can not redirect a new input PIPE when the component is running."
      # Release the original input PIPE
      self.__inPIPE.release_out(password=self.__inPassword)
    # Lock out port of this input PIPE.
    self.__inPIPE = inPIPE
    self.__inPassword = inPIPE.lock_out() # Lock the output port of PIPE
    # Decide the input main key
    if iKey != None:
      self.iKey = iKey

  def start(self,inPIPE:PIPE=None,iKey=None):
    '''
    Start running a process to handle Packets in inPIPE.
    '''
    # If this is a silent component
    if self.coreProcess is None:
      if inPIPE is None:
        if self.__inPIPE is None:
          raise Exception(f"{self.name}: Please give the input PIPE.")
        else:
          # If input PIPE has been linked
          inPIPE = self.__inPIPE
      else:
        # Link (or redirect) the input PIPE
        self.link(inPIPE,iKey)
      # Activate the output PIPE
      self.__outPIPE.activate()
      # Run core process
      if inPIPE.state_is_(mark.silent):
        inPIPE.activate()
      self.__coreProcess = self._start() #self._start(inPIPE=inPIPE,iKey=self.ikey,outPIPE=self.outPIPE,oKey=self.okey)
      assert isinstance(self.__coreProcess,multiprocessing.Process), \
        f"{self.name}: The function _start must return a multiprocessing.Process object!"
    
    # If this is not silent component
    elif self.coreProcess.is_alive():
      # If this component is stranded
      if self.__outPIPE.state_is_(mark.stranded):
        ## If do not need to redirect
        if inPIPE is None or inPIPE.objid == self.__inPIPE.objid:
          self.__inPIPE.restart()
          self.__outPIPE.restart()
        ## If need to redirect input PIPE
        else:
          # Close the core process
          self.__redirect_flag.value = True
          self.__coreProcess.join()
          self.__redirect_flag.value = False
          # Link the new input PIPE
          self.link(inPIPE,iKey)
          # Activate
          self.__outPIPE.restart()
          # Run core process
          if inPIPE.state_is_(mark.silent):
            inPIPE.activate()
          self.__coreProcess = self._start() #self._start(inPIPE=inPIPE,iKey=self.ikey,outPIPE=self.outPIPE,oKey=self.okey)
          assert isinstance(self.__coreProcess,multiprocessing.Process), \
            f"{self.name}: The function _start must return a multiprocessing.Process object!"
    else:
      raise Exception(f"{self.name}: Can only start a silent or restart a stranded Component.")

  def decide_state(self):
    # If input PIPE and outPIPE are the same state
    if self.inPIPE.state == self.outPIPE.state:
      return mark.inPIPE, self.inPIPE.state
    # Else
    else:
      # if input -> active, output -> positive
      if self.inPIPE.timestamp > self.outPIPE.timestamp:
        if self.inPIPE.state_is_(mark.active):
          self.outPIPE.activate()
        elif self.inPIPE.state_is_(mark.wrong):
          self.outPIPE.kill()
        elif self.inPIPE.state_is_(mark.stranded):
          self.outPIPE.pause()
        else:
          # Do not terminated the out PIPE
          # Because it is possible to append data in the out PIPE
          pass
        return mark.inPIPE, self.inPIPE.state
      else:
        if self.outPIPE.state_is_(mark.active):
          self.inPIPE.activate()
        elif self.outPIPE.state_is_(mark.wrong):
          self.inPIPE.kill()
        elif self.outPIPE.state_is_(mark.stranded):
          self.inPIPE.pause()
        else:
          # Stop the input PIPE instantly
          self.inPIPE.stop()
        return mark.outPIPE, self.outPIPE.state

  # Please implement it
  # This function must return a multiprocessing.Process object.
  def _start(self)->multiprocessing.Process:
    raise Exception(f"{self.name}: Please implement the ._start function.")

  def stop(self):
    '''
    Terminate this component normally.
    Note that we do not terminate the core process by this function.
    We hope the core process can be terminated with a mild way.
    '''
    # Stop input PIPE
    assert self.__inPIPE is not None
    self.__inPIPE.stop()
  
  def kill(self):
    '''
    Terminate this component with state: wrong.
    It means errors occurred somewhere.
    Note that we do not kill the core thread by this function.
    We hope the core thread can be terminated with a mild way.
    '''
    # Kill input PIPE
    assert self.__inPIPE is not None
    self.__inPIPE.kill()
  
  def pause(self):
    '''
    Pause the Componnent
    '''
    # Kill input PIPE
    assert self.__inPIPE is not None
    self.__inPIPE.pause()

  def wait(self):
    '''
    Wait until the core thread is finished.
    '''
    if self.__coreProcess is None:
      raise Exception(f"{self.name}: Component has not been started.")
    else:
      # self.__coreProcess.join()
      while not self.__core_process_over.value:
        time.sleep(info.TIMESCALE)

  def get_packet(self):
    '''
    Get packet from input PIPE.
    '''
    assert self.__inPIPE is not None
    return self.__inPIPE.get(password=self.__inPassword)
  
  def put_packet(self,packet):
    self.__outPIPE.put(packet,password=self.__outPassword)

  def create_process(self,target,args=()):
    p = multiprocessing.Process(target=target,args=args,kwargs=(("overFlag",self.__core_process_over),))
    p.daemon = True
    return p

class Chain(ExKaldiRTBase):
  '''
  Chain is a container to easily manage the sequential Component-PIPEs.
  '''
  def __init__(self,name=None):
    # Initial state and name
    super().__init__(name=name)
    # A container to hold components
    self.__chain = []
    self.__inPIPEs = []
    self.__outPIPEs = []
    # Component name -> Index
    self.__name2id = {}
    self.__id = 0

  def add(self,component,inPIPE=None):
    '''
    Add a new component.
    '''
    # Verify chain's state
    for pipe in self.outPIPE:
      if not pipe.state_is_(mark.silent):
        raise f"{self.name}: Chain has been activated. Can not add new components."

    assert isinstance(component,(Component,Joint))
    # If input PIPE is not specified
    if inPIPE is None:
      # Component
      if isinstance(component,Component):
        # if this component has already been linked to a PIPE
        if component.inPIPE is not None:
          # Backup inPIPE
          self.__inPIPEs.append( component.inPIPE )
          # Backup outPIPE
          self.__outPIPEs.append( component.outPIPE )
        # Or find an avaliable PIPE automatically
        else:
          if len(self.__outPIPEs) == 0:
            raise Exception(f"We expect this component should be linked an input PIPE in advance: {component.name}.")
          else:
            assert len(self.__outPIPEs) == 1, \
              f"More than one output port was found in chain. Please specify the input PIPE of this component: {component.name}."
            component.link( self.__outPIPEs[0] )
            self.__outPIPEs[0] = component.outPIPE
      # Joint
      else:
        raise Exception("Joint will be developed in the future version.")
    # Or the input PIPE is specified
    else:
      # Component
      if isinstance(component,Component):
        try:
          self.__outPIPEs.remove( inPIPE )
        except ValueError:
          # If this inPIPE is an outside PIPE, take a backup for it
          self.__inPIPEs.append( inPIPE )
        else:
          # or this is an inside PIPE, remove it from the backup
          pass
        component.link( inPIPE )
        self.__outPIPEs.append( component.outPIPE )
      # Joint
      else:
        raise Exception("Joint will be developed in the future version.")
    
    # Storage and numbering this component 
    self.__chain.append( component )
    self.__name2id[ component.name ] = self.__id
    self.__id += 1

  def start(self):
    # 
    assert len(self.__chain) > 0
    for pipe in self.__outPIPEs:
      assert pipe.state_is_(mark.silent) or pipe.state_is_(mark.stranded)
    # Run all components and joints
    for comp in self.__chain:
      comp.start()
    
  def stop(self):
    assert len(self.__chain) > 0
    # Stop 
    for inPIPE in self.__inPIPEs:
      if inPIPE.state_is_(mark.active) or inPIPE.state_is_(mark.stranded):
        inPIPE.stop()
  
  def kill(self):
    assert len(self.__chain) > 0
    # Stop 
    for comp in self.__chain:
      comp.kill()

  def pause(self):
    assert len(self.__chain) > 0
    # Stop 
    for inPIPE in self.__inPIPEs:
      if inPIPE.state_is_(mark.active):
        inPIPE.pause()

  def wait(self):
    assert len(self.__chain) > 0
    for comp in self.__chain:
      comp.join()

  @property
  def outPIPE(self):
    return self.__outPIPEs[0] if len(self.__outPIPEs) == 1 else self.__outPIPEs

  def component(self,name=None,ID=None)->Component:
    '''
    Get the component or joint by calling its name.
    
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

  def reset(self):
    '''
    Reset the Chain.
    We will reset all components in it.
    '''
    for pipe in self.__outPIPEs:
      assert not pipe.state_is_(mark.active,mark.stranded)
    # Reset all components
    for comp in self.__chain:
      comp.reset()
    # Reset State

############################################
# Joint is designed to merge or seperate pipeline 
# so that the chain can execute multiple tasks
############################################

class Joint(ExKaldiRTBase):
  pass

"""
class Joint(ExKaldiRTBase):

  def __init__(self,outNums=1,name=None):
    super().__init__(name=name)
    assert isinstance(outNums,int) and outNums > 0
    # Defibe input and output PIPEs
    self.__inPIPEs = []
    self.__inPasswords = []
    self.__outPIPEs = []
    self.__outPasswords = []
    for i in range(outNums):
      self.__outPIPEs.append( PIPE(name=f"{i}th output PIPE of "+self.name) )
      self.__outPasswords.append( self.__outPIPEs[i].lock_in() )
    # Each Joint has a core process to run a function to handle packets.
    self.__coreProcess = None
    # ID counter
    self.__id_counter = 0
    # 
    self.__redirect_flag = RawValue(ctypes.c_uint, mark.false)

  @property
  def __id_count(self):
    self.__id_counter += 1
    return self.__id_counter - 1
  
  def reset(self):
    '''
    Clear and reset Joint.
    '''
    assert not (self.state_is_(mark.active) or self.state_is_(mark.stranded)), \
          f"{self.name}: Can not reset a active or stranded Component."
    if self.__coreProcess is not None:
      self.__coreProcess.join()
    self.__coreProcess = None
    for outPIPE in self.__outPIPEs:
      outPIPE.reset()
    self.shift_state_to_(mark.silent)

  @property
  def coreProcess(self)->multiprocessing.Process:
    '''
    Get the core process.
    '''
    return self.__coreProcess

  @property
  def outPIPEs(self)->PIPE:
    return self.__outPIPEs
  
  def link(self,inPIPEs):
    assert self.state_is_(mark.silent)
    assert isinstance(inPIPEs,(list,tuple))
    
    for inPIPE in inPIPEs:
      assert isinstance(inPIPE,PIPE)
      self.__inPIPEs.append( inPIPE )
      self.__inPasswords.append( inPIPE.lock_out() ) # Lock the output port of PIPE

  def start(self,inPIPEs=None):
    '''
    Start running a process to handle Packets in inPIPEs.
    '''
    # If this is a silent Joint
    if self.state_is_(mark.silent):
      # If input PIPE has been linked
      if inPIPEs is None:
        if len(self.__inPIPEs) == 0:
          raise Exception(f"{self.name}: Please give the input PIPEs.")
        else:
          inPIPEs = self.__inPIPEs
      else:
        # Link (or redirect ) the input PIPE
        self.link(inPIPEs)
      # Activate this Joint
      self.shift_state_to_(mark.active)
      # Run core process
      self.__coreProcess = self._start(inPIPEs=inPIPEs)
      assert isinstance(self.__coreProcess,multiprocessing.Process), \
        f"{self.name}: The function _start must return a multiprocessing.Process object!"
    # If this is a stranded pipe
    elif self.state_is_(mark.stranded):
      ## If do not need to redirect
      if inPIPEs is None or inPIPEs == self.__inPIPEs:
        for outPIPE in self.__outPIPEs:
          outPIPE.shift_state_to_(mark.active)
        self.shift_state_to_(mark.active)
      ## If need to redirect input PIPE
      else:
        # Close the core process
        self.__redirect_flag.value = mark.true
        self.__coreProcess.join()
        self.__redirect_flag.value = mark.false
        # Link the new input PIPE
        self.shift_state_to_(mark.silent)
        self.link(inPIPEs)
        # Activate
        for outPIPE in self.__outPIPEs:
          outPIPE.shift_state_to_(mark.active)
        self.shift_state_to_(mark.active)
        # Run core process
        self.__coreProcess = self._start(inPIPEs=inPIPEs)
        assert isinstance(self.__coreProcess,multiprocessing.Process), \
          f"{self.name}: The function _start must return a multiprocessing.Process object!"
    else:
      raise Exception(f"{self.name}: Can only start a silent or restart a stranded Component.")

  # Please implement it
  # This function must return a multiprocessing.Process object.
  def _start(self,inPIPEs)->multiprocessing.Process:
    raise Exception(f"{self.name}: Please implement the ._start function.")

  def stop(self):
    '''
    Terminate this component normally.
    Note that we do not terminate the core process by this function.
    We hope the core process can be terminated with a mild way.
    '''
    # Change state
    self.shift_state_to_(mark.terminated)
    # Append an ENDPOINT flag into output PIPE then terminate it
    for i in range(len(self.__outPIPEs)):
      self.put_packet( i, ENDPOINT )
      self.__outPIPEs[i].stop()
    if len(self.__inPIPEs) > 0:
      for pipe in self.__inPIPEs:
        pipe.stop()
  
  def kill(self):
    '''
    Terminate this component with state: wrong.
    It means errors occurred somewhere.
    Note that we do not kill the core thread by this function.
    We hope the core thread can be terminated with a mild way.
    '''
    # Change state
    self.shift_state_to_(mark.wrong)
    # Kill output PIPE in order to pass error state to the succeeding components
    for outPIPE in self.__outPIPEs:
      outPIPE.kill()
    if len(self.__inPIPEs) > 0:
      for pipe in self.__inPIPEs:
        pipe.kill()
  
  def pause(self):
    '''
    Pause the Componnent
    '''
    self.shift_state_to_(mark.stranded)
    for outPIPE in self.__outPIPEs:
      outPIPE.pause()
    if len(self.__inPIPEs) > 0:
      for pipe in self.__inPIPEs:
        pipe.kill()
        
  def wait(self):
    '''
    Wait until the core thread is finished.
    '''
    if self.__coreProcess is None:
      raise Exception(f"{self.name}: Component has not been started.")
    else:
      self.__coreProcess.join()

  def get_packet(self,inPIPEID):
    assert isinstance(inPIPEID,int) and 0 <= inPIPEID < len(self.__inPIPEs)
    self.__inPIPEs[inPIPEID].get(password=self.__inPasswords[inPIPEID])
  
  def put_packet(self,outPIPEID,packet):
    assert isinstance(outPIPEID,int) and 0 <= outPIPEID < len(self.__outPIPEs)
    self.__outPIPEs[outPIPEID].put(packet,password=self.__inPasswords[outPIPEID])

"""

def dynamic_display(pipe,mapFunc=None):
  '''
  This is a tool for debug or testing.
  '''
  assert isinstance(pipe,PIPE), "<pipe> should be a PIPE object."
  assert not pipe.is_outlocked(), "The out port of <pipe> is locked. Please release it firstly."
  assert not pipe.state_is_(mark.silent), "<pipe> is not activated."
  if pipe.state_is_(mark.stranded):
    print_( "Warning: the PIPE is stranded!" )

  def default_function(pac):
    out = []
    for key,value in pac.items():
      if isinstance(value,np.ndarray):
        temp = " ".join( [ str(v) for v in value[:10] ] )
        out.append( f"{key}: [ {temp} ...] " )
      else:
        out.append( f"{key}: {value} " )
    out = "\n".join(out)
    print_(out)

  if mapFunc is None:
    mapFunc = default_function
  else:
    assert callable( mapFunc )

  # active, stranded, wrong, terminated
  timecost = 0
  while True:
    if pipe.state_is_(mark.active):
      if pipe.is_empty():
        time.sleep( info.TIMESCALE )
        timecost += info.TIMESCALE
        if timecost > info.TIMEOUT:
          raise Exception( f"{pipe.name}: Time out!" )
        continue
      else:
        #print( "debug:", pipe.is_outlocked()  )
        packet = pipe.get()
    elif pipe.state_is_(mark.stranded):
      time.sleep( info.TIMESCALE )
      continue
    else:
      if pipe.is_empty():
        break
      else:
        #print( "debug:", pipe.is_outlocked()  )
        packet = pipe.get()
    
    if is_endpoint( packet ):
      print_(f"----- Endpoint -----")
      continue
    else:
      mapFunc( packet )

  lastState = "terminated" if pipe.state_is_(mark.terminated) else "wrong"
  print_( f"Final state of this PIPE: {lastState} \n Time report: {pipe.report_time()}" )

def dynamic_run(target,inPIPE=None,items=["data"]):
  print_("exkaldirt.base.dynamic_run has been removed in current version. See also exkaldirt.base.dynamic_display function.")

