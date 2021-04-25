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
import ctypes
import time
import random
import datetime
from collections import namedtuple
import glob

from exkaldirt.version import version
from exkaldirt.utils import *

""" from version import version
from utils import *
 """
class Info:
  '''
  A object to hold some configs of ExKaldi-RT.
  '''
  def __init__(self):
    self.__timeout = 1800
    self.__timescale = 0.01
    self.__max_socket_buffer_size = 10000
    # Check Kaldi root directory and ExKaldi-RT tool directory
    self.__find_ctool_root()
    # Get the float floor
    self.__epsilon = self.__get_floot_floor()

  def __find_ctool_root(self):
    '''Look for the ExKaldi-RT C++ command root path.'''
    self.__kaldi_root = None
    if "KALDI_ROOT" in os.environ.keys():
      self.__kaldi_root = os.environ["KALDI_ROOT"]
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
        # out is a string like "/yourhome/kaldi/src/bin/copy-matrix"
        self.__kaldi_root = os.path.dirname( os.path.dirname( os.path.dirname(out)) )

    if self.__kaldi_root is None:
      self.__cmdroot = None
    else:
      decoder = glob.glob( os.path.join(self.__kaldi_root,"src","exkaldirtcbin","exkaldi-online-decoder") )
      tools = glob.glob( os.path.join(self.__kaldi_root,"src","exkaldirtcbin","cutils.*.so") )
      if len(decoder) == 0 or len(tools) == 0:
        print("Warning: ExKaldi-RT C++ source files have not been compiled sucessfully. " + \
              "Please consult the Installation in github: https://github.com/wangyu09/exkaldi-rt ." + \
              "Otherwise, the exkaldi.feature and exkaldi.decode modules are not available."
            )
        self.__cmdroot = None
      else:
        self.__cmdroot = os.path.join(self.__kaldi_root,"src","exkaldirtcbin")

  def __get_floot_floor(self):
    '''Get the floot floor value.'''
    if self.__cmdroot is None:
      return 1.19209e-07
    else:
      sys.path.append( self.__cmdroot )
      try:
        import cutils
      except ModuleNotFoundError:
        raise Exception("ExKaldi-RT Pybind library have not been compiled sucessfully. " + \
                        "Please consult the Installation in github: https://github.com/wangyu09/exkaldi-rt .")
      return cutils.get_float_floor()

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
    '''
    Reset maximum socket buffer size.
    '''
    assert isinstance(size,int) and size > 4
    self.__max_socket_buffer_size = size

  def set_TIMEOUT(self,value):
    '''
    Reset global timeout value.

    Args:
      _value_: a positive int value.
    '''
    assert isinstance(value,int) and value > 0, "TIMEOUT must be an int value."
    self.__timeout = value
  
  def set_TIMESCALE(self,value):
    '''
    Reset global timescale value.

    Args:
      _value_: a float value in (0, 1).
    '''
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
      
    self.__name = name

  @property
  def objid(self):
    '''
    Get the unique object ID.
    '''
    return self.__objid

  @property
  def name(self):
    '''
    Get the name: basename + [ the object id ] .
    '''
    return self.__name + f"[{self.__objid}]"

  @property
  def basename(self):
    '''
    The basename.
    '''
    return self.__name

########################################

class Packet:
  '''
  Packet object is used to hold various stream data, such as audio stream, feature and probability.
  These data will be processed by Components and Joints, and passed in PIPE.
  A packet can hold 4 types of data:
  1, element: int, float, numpy int, numpy float.
  2, vector: numpy 1-d array.
  3, matrix: numpy 2-d array.
  4, text: str.
  '''
  def __init__(self,items,cid,idmaker,mainKey=None):
    '''
    Args:
      _items_: a dict object.
      _cid_: the packet ID. It will be numbered by some components.
      _idmaker_: the object ID of a component who numbered this packet.
      _mainKey_: specify a key as main key. If None, we will decide it automatically.
    '''
    assert isinstance(items,dict), f"<items> must be a dict object but got: {type(items)}."
    self.__data = {}
    # Set items
    if mainKey is not None:
      assert mainKey in items.keys(), f"<mainKey> is not in items: {mainKey}."
      self.__mainKey = mainKey
      for key,value in items.items():
        self.add(key,value)
    else:
      self.__mainKey = None
      for key,value in items.items():
        self.add(key,value)
        if self.__mainKey is None:
          self.__mainKey = key
    # Set chunk id
    assert isinstance(cid,int) and cid >= 0, f"<cid> must be an int value and >= 0 but got: {cid}"
    self.__cid = cid
    assert isinstance(idmaker,int), f"<idmaker> must be an avaliable object ID but got: {idmaker}"
    self.__idmaker = idmaker

  @property
  def cid(self):
    return self.__cid
  
  @property
  def idmaker(self):
    return self.__idmaker

  @property
  def mainKey(self):
    return self.__mainKey
  
  def __getitem__(self,key=None):
    assert key in self.__data.keys()
    return self.__data[key]
  
  def add(self,key:str,data,asMainKey:bool=False):
    '''
    Add one record to packet.
    If this key has already existed, replace the record in packet, otherwise append this new record. 

    Args:
      _key_: (str) key.
      _data_: (int,float,np.int,np.float,np.ndarray,str) value.
      _asMainKey_: (bool) If True, set this key as main key.
    '''
    # Verify key name
    assert isinstance(key,str), f"<key> must be a string but got: {key}."
    assert " " not in key and key.strip() != "", f"<key> can not include space: {key}."
    # Verify value
    if isinstance(data,int):
        data = np.int16(data)
    elif isinstance(data,float):
        data = np.float32(data)
    elif isinstance(data,(np.signedinteger,np.floating)):
        pass
    elif isinstance(data,np.ndarray):
        assert len(data.shape) in [1,2] 
        assert 0 not in data.shape, "Invalid data."
        data = data.copy()
    elif isinstance(data,str):
      assert data != ""
    else:
        raise Exception(f"Packet data can be int, float, str or np.ndarray but got unsupported data type: {type(data)}.")
    self.__data[key] = data
    if asMainKey:
      self.__mainKey = key
  
  def encode(self)->bytes:
    '''
    Encode packet to bytes.

    Return:
      (bytes) encoded object.
    '''
    result = b""
    
    #Encode class name
    result += ( self.__class__.__name__.encode() + b" " )

    # Encode idmaker and fid
    result += uint_to_bytes(self.idmaker, length=4)
    result += uint_to_bytes(self.cid, length=4)

    # If this is not an empty packet
    if self.mainKey is not None:

      # Encode main key
      result += (self.mainKey.encode() + b" ")
      
      # Encode data
      for key,value in self.__data.items():
        # Encode key
        result += key.encode() + b" "
        if isinstance( value,(np.signedinteger,np.floating) ):
          bvalue = element_to_bytes( value )
          flag = b"E"
        elif isinstance(value,np.ndarray):
          if len(value.shape) == 1:
            bvalue = vector_to_bytes( value )
            flag = b"V"
          else:
            bvalue = matrix_to_bytes( value )
            flag = b"M"
        elif isinstance(value,str):
          bvalue = value.encode()
          flag = b"S"
        else:
          raise Exception(f"Packet data can be int, float, str or np.ndarray but got unsupported data type: {type(value)}.")
        result += ( flag + uint_to_bytes( len(bvalue),length=4 ) + bvalue )
  
    return result

  @classmethod
  def decode(cls,bstr):
    '''
    Restorage a packet object fro bytes object.

    Args:
      _bstr_: (bytes) Encoded packet object.
    
    Return:
      (Packet,Endpoint) Decoded object.
    '''
    try:
      with BytesIO(bstr) as sp:
        
        # Read class name
        className = read_string( sp )
        assert className in ["Packet","Endpoint"]

        # Read chunk ID
        idmaker = uint_from_bytes( sp.read(4) )
        cid = uint_from_bytes( sp.read(4) )
        # Read main key
        mainKey = read_string( sp )

        result = {}
        # If this is not an empty packet
        if mainKey != "":
          # Read data
          while True:
            key = read_string(sp)
            if key == "":
              break
            flag = sp.read(1).decode()
            if flag == "E":
              size = uint_from_bytes( sp.read(4) )
              data = element_from_bytes( sp.read(size) )
            elif flag == "V":
              size = uint_from_bytes( sp.read(4) )
              data = vector_from_bytes( sp.read(size) )
            elif flag == "M":
              size = uint_from_bytes( sp.read(4) )
              data = matrix_from_bytes( sp.read(size) )
            elif flag == "S":
              size = uint_from_bytes( sp.read(4) )
              data = sp.read(size).decode()
            else:
              raise Exception(f"Data flag can be E(element), V(vector), M(matrix), S(string) but got a unknown flag: {flag}")

            result[ key ] = data

        # otherwise, this is an empty packet
        else:
          mainKey = None
    
    except Exception as e:
      print_( f"Failed to decode packet. This may be not a bytes packet." )
      raise e

    return globals()[className](items=result,cid=cid,idmaker=idmaker,mainKey=mainKey)
  
  def keys(self):
    return self.__data.keys()
  
  def values(self):
    return self.__data.values()
  
  def items(self):
    return self.__data.items()

  def is_empty(self)->bool:
    '''
    Return "True" if packet is empty.
    '''
    return len(self.keys()) == 0

  def save(self,fileName)->str:
    '''
    Save packet to a binary file.
    
    Args:
      _fileName_: (str) file name with suffix ".pak".
    
    Return:
      (str) finally saved absolute file path.
    '''
    assert isinstance(fileName,str) and len(fileName.strip()) > 0
    fileName = fileName.strip()
    if not fileName.endswith(".pak"):
      fileName += ".pak"
    absname = os.path.abspath(fileName)
    ddir = os.path.dirname(absname)
    if not os.path.isdir(ddir):
      os.makedirs(ddir)
    with open(fileName,"wb") as fw:
      fw.write( self.encode() )
    
    return absname
    
  @classmethod
  def load(cls,fileName):
    '''
    Load packet from binary file.

    Args:
      _fileName_: (str) file name.
    
    Return:
      (Packet,Endpoint) object.
    '''
    assert os.path.isfile(fileName), f"No such file: {fileName}."
    with open(fileName,"rb") as fr:
      data = fr.read()
    return cls.decode(data)

# ENDPOINT is a special packet.
class Endpoint(Packet):
  '''
  A special packet to mark endpoint of audio.
  An endpoint packet is defaultly empty.
  But it also can hold data.
  '''
  def __init__(self,cid,idmaker,items={},mainKey=None):
    super().__init__(items,cid,idmaker,mainKey)
  
def is_endpoint(obj)->bool:
  '''
  If this is Endpoint, return True.
  '''
  return isinstance(obj, Endpoint)

# Standerd output lock
stdout_lock = threading.Lock()

def print_(*args,**kwargs):
  '''
  Print function with thread lock.
  '''
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
  PIPE is used to connect Components (and Joints) and pass Packets.
  It is a Last-In-Last-Out queue.
  It is designed to exchange data and state between mutiple threads.

  A PIPE has five kinds of states:
  | State name | description | Which states can be changed to ? | Is "put" avaliable ? | Is "get" avaliable ?
  1. silent: PIPE is unavaliable. Next State -> active. Put -> Yes, Get -> No.
  2. active: PIPE is working normally. Next State -> stranded, wrong, terminated. Put -> Yes, Get -> Yes.
  3. stranded: PIPE is paused. Next State -> active, wrong, terminated. Put -> No, Get -> No.
  4. wrong: PIPE is finished with a error state. Next State -> silent. Put -> No, Get -> Yes.
  5. terminated: PIPE is finished with a normal state. Next State -> silent. Put -> No, Get -> Yes.

  Note that we will forcely:
  1. remove continuous empty Endpoint flags (only keep the first one).
  2. discard the head packet if it is Endpoint flag.
  '''
  def __init__(self,name=None):
    '''
    Args:
      _name_: (str) If None, we will name it with the class name.
    '''
    # Initilize state and name
    super().__init__(name=name)
    # Set a cache to pass data
    self.__cache = queue.Queue()
    self.__cacheSize = 0
    # Flags used to communicate between different components
    self.__state = mark.silent
    self.__inlocked = False
    self.__outlocked = False
    self.__last_added_endpoint = False
    self.__firstPut = 0.0
    self.__lastPut = 0.0
    self.__firstGet = 0.0
    self.__lastGet = 0.0
    self.__lastID = (-1,-1)
    self.__time_stamp = time.time()
    # Password to access this PIPE
    self.__password = random.randint(0,100)
    # Class backs functions
    self.__callbacks = []

  def state_is_(self,*m)->bool:
    '''
    args:
      _m_: (mark symbol or symbols).
    '''
    return self.__state in m

  def __shift_state_to_(self,m):
    assert m in mark
    self.__state = m
    self.__time_stamp = time.time()

  @property
  def state(self)->int:
    return self.__state

  @property
  def timestamp(self):
    '''
    Return:
      (time.time) The lastest time when state is changed.
    '''
    return self.__time_stamp

  #############
  # Lock input or output port
  #############

  def is_inlocked(self)->bool:
    '''
    If input port is locked, return "True".
    '''
    return self.__inlocked
  
  def is_outlocked(self)->bool:
    '''
    If output port is locked, return "True".
    '''
    return self.__outlocked

  def lock_in(self)->int:
    '''
    Lock this input of PIPE and return the password.
    If input port had already been locked by another master, do nothing and return None.

    Return:
      (int,None) Password.
    '''
    if self.is_inlocked():
      return None
    self.__inlocked = True
    return self.__password
  
  def lock_out(self)->int:
    '''
    Lock this output of PIPE and return the password.
    If output port had already been locked by another master, do nothing and return None.

    Return:
      (int,None) Password.
    '''
    if self.is_outlocked():
      return None
    self.__outlocked = True
    return self.__password

  def release_in(self,password:int):
    '''
    Release the input port.

    Args:
      _password_: (int) password.
    '''    
    if self.is_inlocked:
      if password == self.__password:
        self.__inlocked = False
      else:
        print_(f"{self.name}: Wrong password to release input port!")
  
  def release_out(self,password:int):
    '''
    Release the output port.

    Args:
      _password_: (int) password.
    '''      
    if self.is_outlocked:
      if password == self.__password:
        self.__outlocked = False
      else:
        print_(f"{self.name}: Wrong password to release output port!")

  #############
  # Some operations
  #############

  def clear(self):
    '''
    Clear cache.
    '''
    assert not self.state_is_(mark.active), f"{self.name}: Can not clear an active PIPE."
    # Clear
    #size = self.size()
    #for i in range(size):
    #  self.__cache.get()
    self.__cache.queue.clear()
  
  def reset(self):
    '''
    Reset this PIPE when state is wrong or terminated.

    Do:
      1. clear data,
      2. reset state to silent,
      3. reset endpoint and time report.
    Do not:
      1. reset input lock and output lock flags.
      2. reset the callbacks. 
    '''
    if self.state_is_(mark.silent):
      return None

    assert not (self.state_is_(mark.active) or self.state_is_(mark.stranded)), \
          f"{self.name}: Can not reset an active or stranded PIPE."
    # Clear cache
    self.clear()
    # Reset state
    self.__shift_state_to_(mark.silent)
    # A flag to remove continue ENDPOINT or head ENDPOINT 
    self.__last_added_endpoint = False
    # flags to report time points
    self.__firstPut = 0.0
    self.__lastPut = 0.0
    self.__firstGet = 0.0
    self.__lastGet = 0.0

  def activate(self):
    '''
    Activate a silent or stranded PIPE.
    State:  silent, stranded -> active.
    '''
    if not self.state_is_(mark.active):
      assert self.state_is_(mark.silent,mark.stranded), f"{self.name}: can only activate a silent or stranded PIPE but PIPE state is: {self.state}."
      self.__shift_state_to_(mark.active)

  def kill(self):
    '''
    Kill a silent, active or stranded PIPE.
    State: silent, active or strande -> wrong.
    '''
    if not self.state_is_(mark.wrong):
      assert self.state_is_(mark.active,mark.silent,mark.stranded), f"{self.name}: can only kill a silent, stranded or active PIPE but PIPE state is: {self.state}."
      self.__shift_state_to_(mark.wrong)
  
  def stop(self):
    '''
    Stop a silent, active, stranded PIPE.
    State: silent, active or strande -> terminated.
    Append an empty endpoint packet to the tail if necessary.
    '''
    if not self.state_is_(mark.terminated):
      assert self.state_is_(mark.active,mark.silent,mark.stranded), f"{self.name}: can only stop a silent, stranded or active PIPE but PIPE state is: {self.state}."
      # Append a endpoint flag
      if not self.__last_added_endpoint:
        self.__cache.put( Endpoint(cid=self.__lastID[0]+1,idmaker=self.__lastID[1]) )
        self.__cacheSize += 1
        self.__last_added_endpoint = True
      # Shift state
      self.__shift_state_to_(mark.terminated)
  
  def pause(self):
    '''
    pause an active PIPE.
    State: active -> stranded.
    '''
    if not self.state_is_(mark.stranded):
      assert self.state_is_(mark.active), f"{self.name}: Can only pause an active PIPE but the state is: {self.state}."
      self.__shift_state_to_(mark.stranded)

  def size(self)->int:
    '''
    Get the cache size.

    Return:
      (int).
    '''
    return self.__cacheSize

  def is_empty(self)->bool:
    '''
    If there is no any data in PIPE, return True.

    Return:
      (bool).
    '''
    return self.__cacheSize == 0

  def get(self,password=None,timeout=info.TIMEOUT)->Packet:
    '''
    Pop a packet from head.
    Can get packet from: active, wrong, terminated PIPE.
    Can not get packet from: silent and stranded PIPE. 

    Args:
      _password_: (None,int) If PIPEport has been locked, password is necessary to access it.
      _timeout_: (int).
    
    Return:
      (False,Packet,Endpoint) If PIPE is unavaliable, return "False", else return a Packet or Endpoint object.
    '''
    if self.state_is_(mark.silent,mark.stranded):
      print_( f"Warning, {self.name}: Failed to get packet in PIPE. PIPE state is or silent or stranded." )
      return False

    # If output port is locked
    if self.is_outlocked():
      if password is None:
        raise Exception(f"{self.name}: Output of PIPE is clocked. Release or give the password to access it.")
      elif password != self.__password:
        raise Exception(f"{self.name}: Wrong password to access the PIPE.")

    packet = self.__cache.get(timeout=timeout)
    # Record time stamp
    if self.__firstGet == 0.0:
      self.__firstGet = datetime.datetime.now()
    self.__lastGet = datetime.datetime.now()
    self.__cacheSize -= 1

    return packet
  
  def put(self,packet,password=None):
    '''
    Push a new packet to tail.
    Note that: we will remove the continuous empty Endpoint packet.
    Can put packet to: silent, alive (If this is a silent PIPE, activate it automatically).
    Can not put packet to: wrong, terminated and stranded PIPE. 
    
    Args:
      _packet_: (Packet,Endpoint).
      _password_: (None,int) If PIPEport has been locked, password is necessary to access it.
    
    Return:
      (bool) return "True" is put done otherwise "False".
    '''
    if self.state_is_(mark.wrong,mark.terminated,mark.stranded):
      print_( f"{self.name}: Failed to put packet in PIPE. PIPE state is not active or silent." )
      return False

    # If input port is locked
    if self.is_inlocked():
      if password is None:
        raise Exception(f"{self.name}: Input of PIPE is clocked. Release or give the password to access it.")
      elif password != self.__password:
        raise Exception(f"{self.name}: Wrong password to access the PIPE.")

    if self.state_is_(mark.silent):
      self.__shift_state_to_(mark.active)

    assert isinstance(packet,Packet), f"{self.name}: Only Packet can be appended in PIPE."
    
    # record time stamp
    if self.__firstPut == 0.0:
      self.__firstPut = datetime.datetime.now()
    self.__lastPut = datetime.datetime.now()
    # remove endpoint continuous flags and call back 
    if is_endpoint(packet):
      if (not self.__last_added_endpoint) or (not packet.is_empty()):
        self.__cache.put(packet)
        self.__last_added_endpoint = True
        self.__cacheSize += 1
        self.__lastID = (packet.cid,packet.idmaker)
    else:
      self.__cache.put(packet)
      self.__cacheSize += 1
      self.__last_added_endpoint = False
      self.__lastID = (packet.cid,packet.idmaker)
      for func in self.__callbacks:
        func(packet)
    
    return True
  
  def to_list(self,mapFunc=None)->list:
    '''
    Dump all packets to lists divided by endpoint flags.
    Only terminated and wrong PIPE can be converted.

    Args:
      _mapFunc_: (callable function or object) A map function to define how to get the data.
                Defaultly, we only keep the mainKey data.
                For example: mapFunc = lambda packet:packet[packet.mainKey]
    '''
    assert self.state_is_(mark.terminated,mark.wrong), \
          f"{self.name}: Only terminated or wrong PIPE can be dump to list."
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
      if is_endpoint(packet):
        if not packet.is_empty():
          partial.append( mapFunc(packet) )
        if len(partial) > 0:
          result.append( partial )
          partial = []
      elif not packet.is_empty():
        partial.append( mapFunc(packet) )
    if len(partial)>0:
      result.append( partial )

    return result[0] if len(result) == 1 else result

  def report_time(self):
    '''
    Report time information.

    Return:
      (namedtuple) time stamp information if they are reported.
    '''
    keys = ["name",]
    values = [self.name,]
    for name in ["firstPut","lastPut","firstGet","lastGet"]:
      value = getattr(self, f"_{type(self).__name__}__{name}")
      if value != 0.0:
        keys.append(name)
        values.append(value)
    return namedtuple("TimeReport",keys)(*values)

  def callback(self,func):
    '''
    Add a callback function executing when a new packet is appended in PIPE.
    Only allow add a new callback when state is silent.

    Args:
      _func_: (callable object or None) a function or callable object. If None, clear callbacks.
              for example: 
                def call_func(packet):
                  print( packet[packet.mainKey] )
    '''
    assert self.state_is_(mark.silent), f"{self.name}: Only allow add a new callback to a silent PIPE but the state is: {self.state}."
    if func is None:
      self.__callbacks.clear()
    else:
      assert callable(func), f"{self.name}: <func> should be a callable funtion or object but got: {type(func)}."
      self.__callbacks.append( func )

class NullPIPE(PIPE):
  '''
  NullPIPE is a special PIPE.
  It can not storage packets but can pass state information.
  '''
  def __init__(self,name=None):
    super().__init__(name=name)

  def clear(self):
    return None
    
  def reset(self):
    if self.state_is_(mark.silent):
      return None

    assert not self.state_is_(mark.active,mark.stranded), \
          f"{self.name}: Can not reset a active or stranded PIPE."
    # Reset state
    self.__shift_state_to_(mark.silent)
    # A flag to remove continue ENDPOINT or head ENDPOINT 
  
  def size(self):
    return 0

  def is_empty(self)->bool:
    return True

  def get(self,password=None,timeout=info.TIMEOUT)->Packet:
    raise Exception(f"{self.name}: Null PIPE can not return packet.")
  
  def put(self,packet,password=None):
    raise Exception(f"{self.name}: Null PIPE can not storage packet.")
  
  def to_list(self,mapFunc=None)->list:
    raise Exception(f"{self.name}: Null PIPE can not convert to list.")

  def report_time(self):
    raise Exception(f"{self.name}: Null PIPE can not report time info.")

  def callback(self,func):
    raise Exception(f"{self.name}: Null PIPE can not add callback functions.")

def is_nullpipe(pipe):
  '''
  If this is Endpoint, return "True".
  '''
  return isinstance(pipe,NullPIPE)

class Component(ExKaldiRTBase):
  '''
  Components are used to process Packets.
  Components can only link to one input PIPE and has one output PIPE.
  
  Some components would generate new packets and renumber them (most components in exkaldirt.stream module are this type.)
  Some components would not generate new packets. They will directly add new data in old packet and passed it to next PIPE.
  '''
  def __init__(self,oKey="data",name=None):
    '''
    Args:
      _oKey_: (str or list/tuple of strs), the output key(s) name of new generated data. 
    '''
    # Initial state and name
    super().__init__(name=name)
    # Define input and output PIPE
    # Input PIPE need to be linked
    self.__inPIPE = None
    self.__inPassword = None
    self.__outPIPE = PIPE(name=f"The output PIPE of "+self.basename)
    self.__outPassword = self.__outPIPE.lock_in() # Lock the in-port of output PIPE
    # Each component has a core process to run a function to handle packets.
    self.__coreThread = None
    # If need to redirect the input PIPE
    # We will stop the core process firstly and then link a new input PIPE and restart core process.
    self.__redirect_flag = False
    # process over flag
    self.__core_thread_over = False
    # The key
    if isinstance(oKey,str):
      self.__oKey = (oKey,)
    else:
      assert isinstance(oKey,(tuple,list)), f"{self.name}: <oKey> must be a string or list/tuple of strings but got: {oKey}."
      for i in oKey:
        assert isinstance(i,str), f"{self.name}: <oKey> must be a string or list/tuple of strings but got: {oKey}."
      self.__oKey = tuple(oKey)
    self.__iKey = None

  @property
  def iKey(self):
    return self.__iKey
  
  @property
  def oKey(self):
    return self.__oKey

  def reset(self):
    '''
    Reset component.
    '''
    if self.coreThread is None:
      return None
    elif self.coreThread.is_alive():
      raise Exception(f"{self.name}: Component is active and can not reset. Please stop it firstly.")
    else:
      self.__coreThread = None
      self.__outPIPE.reset()
      if not self.__inPIPE.state_is_(mark.silent):
        self.__inPIPE.reset()
      self.__core_thread_over = False
      self.__redirect_flag = False

  @property
  def coreThread(self)->threading.Thread:
    '''
    Get the core process.
    '''
    return self.__coreThread

  @property
  def inPIPE(self)->PIPE:
    return self.__inPIPE

  @property
  def outPIPE(self)->PIPE:
    return self.__outPIPE
  
  def link(self,inPIPE:PIPE,iKey=None):
    '''
    Link an input PIPE.
    State must be silent.

    Args:
      _inPIPE_: (PIPE) input PIPE.
      _iKey_: (None,str) specify the input key. If None, automatically find and use the mainKey of input packet.
    '''
    assert isinstance(inPIPE,PIPE), f"{self.name}: Can only link a PIPE (or NullPIPE) but got: {type(inPIPE)}."

    if iKey is not None:
      assert isinstance(iKey,str), f"{self.name}: <iKey> should be s string but got: {iKey}."
      self.__iKey = iKey

    # check if redirection is required
    if self.coreThread is not None:
      assert not self.coreThread.is_alive(), f"{self.name}: Can not redirect a new input PIPE when the component is running."
      # If it does not need to redict the input PIPE
      if inPIPE == self.__inPIPE:
        return None
    
    if self.__inPIPE is not None:
      self.__inPIPE.release_out(password=self.__inPassword)
    #
    assert not inPIPE.is_outlocked(), "The output port of PIPE has already been locked by another master. Please release it firstly."
    # Lock out port of this input PIPE
    self.__inPIPE = inPIPE
    self.__inPassword = inPIPE.lock_out() # Lock the output port of PIPE

  def start(self,inPIPE:PIPE=None,iKey:str=None):
    '''
    Start running a thread to deal with packets in input PIPE and append th result to output PIPE.

    Args:
      _inPIPE_: (PIPE) the input PIPE. If an PIPE has already linked before, this is not necessary.
      _iKey_: (None,str) in key.
    '''
    assert self.outPIPE.state_is_(mark.silent,mark.stranded), f"{self.name}: Can only start a silent or restart a stranded Component but the state is: {self.outPIPE.state}."

    # If this is a silent component
    if self.coreThread is None:
      if inPIPE is None:
        if self.__inPIPE is None:
          raise Exception(f"{self.name}: Please give the input PIPE.")
        else:
          # If input PIPE has been linked
          inPIPE = self.__inPIPE
      else:
        # Link (or redirect) the input PIPE
        self.link( inPIPE,iKey )
      # Activate the output PIPE
      self.__outPIPE.activate()
      # Try to activate input PIPE
      if inPIPE.state_is_(mark.silent):
        inPIPE.activate()
      # Run core process
      self.__coreThread = self._create_thread(func=self.__core_thread_loop_wrapper)

    # If this is a stranded component
    else:
      ## If do not need to redirect
      if inPIPE is None or inPIPE.objid == self.__inPIPE.objid:
        self.__inPIPE.activate()
        self.__outPIPE.activate()
      ## If need to redirect input PIPE
      else:
        # Close the core process
        self.__redirect_flag = True
        self.wait()
        self.__redirect_flag = False
        # Link the new input PIPE
        self.link(inPIPE,iKey)
        # Activate
        self.__outPIPE.activate()
        # Run core process
        if inPIPE.state_is_(mark.silent):
          inPIPE.activate()
        # Run core process
        self.__coreThread = self._create_thread(func=self.__core_thread_loop_wrapper)

  def _create_thread(self,func)->threading.Thread:
    '''
    Create and start a thread.
    '''
    coreThread = threading.Thread(target=func)
    coreThread.setDaemon(True)
    coreThread.start()
    return coreThread

  def decide_state(self):
    '''
    This function is used in core thread.
    It will check current state and time stamp of input PIPE and output PIPE, 
    then decide the next state.

    Return:
      1, (int,None) inPIPE or outPIPE mark or None. To mark who decided the next state.
      2, (int) state mark. The next state.
    '''
    assert (not self.inPIPE.state_is_(mark.silent)) and  (not self.outPIPE.state_is_(mark.silent)), \
           "Can not decide state because input PIPE or outPIPE has not been activated."

    # If input and output PIPE have the same state
    if self.inPIPE.state == self.outPIPE.state:
      return None, self.inPIPE.state

    # firstly check whether there is wrong state
    # if there is, terminate input and output PIPE instantly
    if self.inPIPE.state_is_(mark.wrong):
      if not self.outPIPE.state_is_(mark.terminated):
        self.outPIPE.kill()
      return mark.inPIPE, mark.wrong

    elif self.outPIPE.state_is_(mark.wrong):
      if not self.inPIPE.state_is_(mark.terminated):
        self.inPIPE.kill()
      return mark.outPIPE, mark.wrong
    
    else:
      #  in state might be: active, stranded, terminated
      # out state might be: active, stranded, terminated
      # and they do not have the same state

      # if output PIPE is terminated
      # terminate input PIPE instantly
      if self.outPIPE.state_is_(mark.terminated):
        if not self.inPIPE.state_is_(mark.active):
          self.inPIPE.stop()
        return mark.outPIPE, mark.terminated
      else:
        #  in state might be: active, terminated, stranded 
        # out state might be: active, stranded
        # and they do not have the same state

        if self.inPIPE.state_is_(mark.active):
          # the output state must be stranded
          # so we now compare the timestamp of in PIPE and out PIPE 
          if self.inPIPE.timestamp > self.outPIPE.timestamp:
            self.outPIPE.activate()
            return mark.inPIPE, mark.active
          else:
            self.inPIPE.pause()
            return mark.outPIPE, mark.stranded

        elif self.inPIPE.state_is_(mark.terminated):
          if self.outPIPE.state_is_(mark.active):
            return mark.inPIPE, mark.terminated
          else:
            return mark.outPIPE, mark.stranded
        else:
          # the output state must be active
          if self.inPIPE.timestamp > self.outPIPE.timestamp:
            self.outPIPE.pause()
            return mark.inPIPE, mark.stranded
          else:
            self.inPIPE.activate()
            return mark.outPIPE, mark.active
 
  def decide_action(self):
    '''
    A standerd function to decide the behavior according to master and next state decided by .decide_state function.
    
    Return:
      (bool) True -> It ok to get a packet from inPIPE.
             False -> Can not get new packet because of error or other reasons.
             None -> All packets are exausted and task is over.
    '''
    timecost = 0

    while True:
      # firstly decide next state
      master, state = self.decide_state()

      # if state is active, check whether it is ready to get packet
      if state == mark.active:
        if self.inPIPE.is_empty():
          time.sleep(info.TIMESCALE)
          timecost += info.TIMESCALE
          if timecost > info.TIMEOUT:
            print(f"{self.name}: Timeout!")
            self.inPIPE.kill()
            self.outPIPE.kill()
            return False
          else:
            continue
        else:
          return True
      # if state is wrong
      elif state == mark.wrong:
        return False
      # if state is stranded
      elif state == mark.stranded:
        time.sleep( info.TIMESCALE )
        if self.__redirect_flag is True:
          break
        continue
      # if state is terminated
      elif state == mark.terminated:
        if master == mark.outPIPE:
          return False
        else:
          if self.inPIPE.is_empty():
            return None
          else:
            return True

  def core_loop(self):
    '''
    The core loop of core thread.
    '''
    raise Exception(f"{self.name}: Please implement the core_loop function.")

  def __core_thread_loop_wrapper(self):
    '''
    A wrapper for core loop function.
    '''
    self.__core_thread_over = False
    print_(f"{self.name}: Start...")
    try:
      self.core_loop()
    except Exception as e:
      if not self.inPIPE.state_is_(mark.wrong,mark.terminated):
        self.inPIPE.kill()
      if not self.outPIPE.state_is_(mark.wrong,mark.terminated):
        self.outPIPE.kill()
      raise e
    else:
      if not self.outPIPE.state_is_(mark.wrong,mark.terminated):
        self.inPIPE.stop()
      if not self.outPIPE.state_is_(mark.wrong,mark.terminated):
        self.outPIPE.stop()
    finally:
      print_(f"{self.name}: Stop!")
      self.__core_thread_over = True

  def stop(self):
    '''
    Terminate this component normally.
    Note that we do not terminate the core thread by this function.
    We hope the core thread can be terminated with a mild way.
    '''
    assert self.coreThread is not None and self.coreThread.is_alive(), f"{self.name}: component is not running."
    self.__inPIPE.stop()

  def kill(self):
    '''
    Terminate this component with state: wrong.
    It means errors occurred somewhere.
    Note that we do not kill the core thread by this function.
    We hope the core thread can be terminated with a mild way.
    '''
    assert self.coreThread is not None and self.coreThread.is_alive(), f"{self.name}: component is not running."
    self.__outPIPE.kill()

  def pause(self):
    '''
    Pause the Component.
    '''
    assert self.coreThread is not None and self.coreThread.is_alive(), f"{self.name}: component is not running."
    self.__inPIPE.pause()

  def wait(self):
    '''
    Wait until the core thread is finished
    '''
    assert self.coreThread is not None, f"{self.name}: component is not running."
    self.__coreThread.join()
    #while not self.__core_thread_over:
    #  time.sleep(info.TIMESCALE)
    #self.__coreThread.terminate()
    #self.__coreThread.join()

  def get_packet(self):
    '''
    Get packet from input PIPE.

    Return:
     (Packet).
    '''
    assert self.__inPIPE is not None, f"{self.name}: component has not been linked to an input PIPE."
    return self.__inPIPE.get(password=self.__inPassword)

  def put_packet(self,packet):
    '''
    Put a packet to output PIPE.
    '''
    self.__outPIPE.put(packet,password=self.__outPassword)

class Chain(ExKaldiRTBase):
  '''
  Chain is a container to easily manage the sequential Component-PIPEs.
  '''
  def __init__(self,name=None):
    # Initial state and name
    super().__init__(name=name)
    # A container to hold components
    self.__chain = []
    self.__inPIPE_Pool = []
    self.__outPIPE_Pool = []
    # Component name -> Index
    self.__name2index = {}
    self.__index = 0

  def add(self,node,inPIPE=None,iKey=None):
    '''
    Append a new component or joint into chain.
    If this node has not been linked to an input PIPE and <inPIPE> is None,
    We will try to link it to the tail of chain automatically. 

    Args:
      _node_: (Component,Joint).
      _inPIPE_: (PIPE) specify the input PIPE.
      _iKey_: (str) specify the input key. 
    '''
    # Verify chain's state
    # Only allow adding new node to a silent chain.
    for pipe in self.__outPIPE_Pool:
      assert pipe.state_is_(mark.silent), f"{self.name}: Chain has been activated. Can not add new nodes."
    assert isinstance(node,(Component,Joint)), f"{self.name}: <node> must be a Component or Joint object but got: {type(node)}."

    # if input PIPE is not specified
    if inPIPE is None:
      # if node is a component
      if isinstance(node,Component):
        # if this component has already been linked to an input PIPE
        if node.inPIPE is not None:
          # if the input PIPE is one of PIPEs in outPIPE pool,
          #    remove the cache and does need to take a backup of this inPIPE
          if node.inPIPE in self.__outPIPE_Pool:
            self.__outPIPE_Pool.remove( node.inPIPE )
          # if the input PIPE is an external PIPE ( not included in the chain ),
          #    we need to add this input PIPE in Pool to take a backup of this PIPE
          else:
            self.__inPIPE_Pool.append( node.inPIPE )
          # storage output PIPE to Pool
          self.__outPIPE_Pool.append( node.outPIPE )
        # if the input PIPE is not been linked. We will try to link it automatically.
        else:
          # if output PIPE pool is empty
          if len(self.__outPIPE_Pool) == 0:
            raise Exception(f"No pipe is avaliable in poll. We expect this component should be linked an input PIPE in advance: {node.name}.")
          # 1.1.2.2 if output PIPE pool is not empty
          else:
            assert len(self.__outPIPE_Pool) == 1, \
              f"More than one output port was found in chain input pool. Please specify the input PIPE of this component: {node.name}."
            node.link( self.__outPIPE_Pool[0], iKey=iKey )
            # take a backup
            self.__outPIPE_Pool[0] = node.outPIPE
      # if node is a joint
      else:
        # if the input PIPE has already been linked
        if node.inNums > 0:
          # for pipe existed in outPIPE Pool, remove it
          # or take a backup
          for pipe in node.inPIPE:
            if pipe in self.__outPIPE_Pool:
              self.__outPIPE_Pool.remove( pipe )
            else:
              self.__inPIPE_Pool.append( pipe )
          for pipe in node.outPIPE:
            self.__outPIPE_Pool.append( pipe )
        else:
          if len(self.__outPIPE_Pool) == 0:
            raise Exception(f"We expect this component should be linked an input PIPE in advance: {node.name}.")
          else:
            node.link( self.__outPIPE_Pool )
            self.__outPIPE_Pool = list( node.outPIPE )
    
    # if the input PIPE is specified
    else:
      # if node is component
      if isinstance(node,Component):
        assert isinstance(inPIPE,PIPE), f"{self.name}: <inPIPE> is not a PIPE object: {type(inPIPE)}."
        # if the input PIPE is one of PIPEs in outPIPE pool,
        #    remove the cache and does need to take a backup of this inPIPE
        # if the node has already been linked to another PIPE, we try to redirect it.
        if node.inPIPE is not None:
          if node.inPIPE != inPIPE:
            print_( f"Warning: Component {node.name} has already been linked to another PIPE. We will try to redirect it." )

        if inPIPE in self.__outPIPE_Pool:
          self.__outPIPE_Pool.remove( inPIPE )
        # if the input PIPE is an external PIPE ( not included in the chain ),
        #     we need to add this input PIPE in Pool to take a backup of this PIPE
        else:
          self.__inPIPE_Pool.append( inPIPE )
        # link input PIPE
        node.link( inPIPE, iKey=iKey )
        # storage output PIPE to Pool
        self.__outPIPE_Pool.append( node.outPIPE )

      # Joint
      else:
        assert isinstance(inPIPE,(tuple,list)), f"{self.name}: <inPIPE> must be a PIPE or list/tuple of PIPEs: {type(inPIPE)}."
        assert len(set(inPIPE)) == len(inPIPE), f"{self.name}: <inPIPE> has repeated PIPE object."
        if node.inNums > 0:
          print_( f"Warning: Joint {node.name} has already been linked to another PIPE. We will try to redirect it." )

        for pipe in inPIPE:
          assert isinstance(pipe, PIPE)
          if pipe in self.__outPIPE_Pool:
            self.__outPIPE_Pool.remove( pipe )
          else:
            self.__inPIPE_Pool.append( pipe )
        
        node.link( inPIPE )
        # storage output pipes
        self.__outPIPE_Pool.extend( node.outPIPE )
    
    # Remove repeated inPIPE and outPIPE in PIPE pool ( keep order )
    tempInPool = []
    for pipe in self.__inPIPE_Pool:
      if pipe not in tempInPool:
        tempInPool.append( pipe )
    self.__inPIPE_Pool = tempInPool

    tempOutPool = []
    for pipe in self.__outPIPE_Pool:
      if pipe not in tempOutPool:
        tempOutPool.append( pipe )
    self.__outPIPE_Pool = tempOutPool

    # Storage and numbering this node 
    self.__chain.append( node )
    self.__name2index[ node.name ] = (node.basename,self.__index)
    self.__index += 1

  def get_node(self,name=None,index=None):
    '''
    Get the component or joint by calling its name or index in chain.
    If multiple nodes have the same name, return the first one.
    
    Args:
      _name_: (str) the name (or basename) of Component.
      _index_: (int) the index number of Component.
    
    Return:
      (Component,Joint).
    '''
    assert not (name is None and index is None), f"{self.name}: Both <name> and <index> are None."

    if name is not None:
      # If this is a name (basename+[objID])
      if name in self.__name2index.keys():
        index = self.__name2index[name]
        return self.__chain[index]
      # ifthis is a basename
      else:
        for basename,index in self.__name2index.values():
          if basename == name:
            return self.__chain[index]
        raise Exception(f"{self.name}: No such Node: {name}")
    else:
      assert isinstance(index,int), f"{self.name}: <index> must be an int value."
      return self.__chain[index]

  def start(self):
    '''
    Run or restart chain.
    '''
    assert len(self.__chain) > 0, f"{self.name}: chain is empty!"
    for pipe in self.__outPIPE_Pool:
      assert pipe.state_is_(mark.silent,mark.stranded), f"{self.name}: can start a silent or stranded chain."
    # Run all components and joints
    for node in self.__chain:
      node.start()
    
  def stop(self):
    '''
    Stop chain.
    '''
    assert len(self.__chain) > 0, f"{self.name}: chain is empty!"
    for pipe in self.__inPIPE_Pool:
      pipe.stop()
  
  def kill(self):
    assert len(self.__chain) > 0, f"{self.name}: chain is empty!"
    # Stop 
    for node in self.__chain:
      node.kill()

  def pause(self):
    assert len(self.__chain) > 0, f"{self.name}: chain is empty!"
    # Stop 
    for pipe in self.__inPIPE_Pool:
      if pipe.state_is_(mark.active):
        pipe.pause()

  def wait(self):
    assert len(self.__chain) > 0, f"{self.name}: chain is empty!"
    for node in self.__chain:
      node.wait()

  @property
  def inPIPE(self):
    return self.__inPIPE_Pool[0] if len(self.__inPIPE_Pool) == 1 else self.__inPIPE_Pool

  @property
  def outPIPE(self):
    return self.__outPIPE_Pool[0] if len(self.__outPIPE_Pool) == 1 else self.__outPIPE_Pool

  def reset(self):
    '''
    Reset the Chain.
    We will reset all components in it.
    '''
    for pipe in self.__outPIPE_Pool:
      assert not pipe.state_is_(mark.active,mark.stranded), f"{self.name}: can not reset an active or stranded chain."
    # Reset all nodes
    for node in self.__chain:
      node.reset()

############################################
# Joint is designed to merge or seperate pipeline 
# so that the chain can execute multiple tasks
############################################

class Joint(ExKaldiRTBase):
  '''
  Joint are used to process Packets.
  Joints can link to multiple input PIPEs and output PIPEs.
  '''
  def __init__(self,jointFunc,outNums=1,name=None):
    '''
    Args:
      _jointFunc_: (callable object) the joint function.
                   for example:
                    def joint_func(items):
                      # items is a list of dict objects from all input PIPEs.
                      do dome process and generate multiple dicts
                      return dict1, dict2 ...
      _outNums_: (int) numbers of output PIPEs.
      _name_: (str).
    '''
    # Initial state and name
    super().__init__(name=name)
    # Define input and output PIPE
    # Input PIPE need to be linked
    self.__inPIPE_Pool = []
    self.__inPassword_Pool = []
    self.__outPIPE_Pool = []
    self.__outPassword_Pool = []
    assert isinstance(outNums,int) and outNums > 0
    self.__inNums = 0
    self.__outNums = outNums
    for i in range(outNums):
      self.__outPIPE_Pool.append( PIPE( name=f"{i}th output PIPE of "+self.basename ) )
      self.__outPassword_Pool.append( self.__outPIPE_Pool[i].lock_in() )  # Lock the in-port of output PIPE
    # Each joint has a core process to run a function to handle packets.
    self.__coreThread = None
    # If need to redirect the input PIPE
    # We will stop the core process firstly and then link a new input PIPE and restart core process.
    self.__redirect_flag = False
    # process over flag. used to terminate core process forcely
    self.__core_thread_over = False
    # define a joint function
    assert callable(jointFunc)
    self.__joint_function = jointFunc

  @property
  def inNums(self):
    return self.__inNums
  
  @property
  def outNums(self):
    return self.__outNums

  def reset(self):
    '''
    Clear and reset joint.
    '''
    if self.coreThread is None:
      return None
    else:
      assert not self.coreThread.is_alive(), f"{self.name}: Joint is active and can not reset. Please stop it firstly."
      self.__coreThread = None
      for pipe in self.__outPIPE_Pool:
        pipe.reset()
      self.__core_thread_over = False

  @property
  def coreThread(self)->threading.Thread:
    '''
    Get the core process.
    '''
    return self.__coreThread

  @property
  def inPIPE(self)->tuple:
    return tuple(self.__inPIPE_Pool)

  @property
  def outPIPE(self)->tuple:
    return tuple(self.__outPIPE_Pool)
  
  def link(self,inPIPE):
    '''
    Add a new inPIPE into input PIPE pool.
    Or replace the input PIPE pool with a list of PIPEs.
    '''
    if self.coreThread is not None:
      assert not self.coreThread.is_alive(), f"{self.name}: Can not redirect a new input PIPE when the joint is running."
    
    # 1. If this is a list/tuple of PIPEs
    if isinstance(inPIPE, (list,tuple)):
      assert len(set(inPIPE)) == len(inPIPE), f"{self.name}: there is repeated PIPE in input PIPEs."
      # 1.1 release the input PIPEs in Pool
      for i in range(self.__inNums):
        self.__inPIPE_Pool[i].release_out(password=self.__inPassword_Pool[i])
      # 1.2 storage new PIPEs
      self.__inPassword_Pool = []
      for pipe in inPIPE:
        assert isinstance(pipe, PIPE)
        self.__inPassword_Pool.append( pipe.lock_out() )
      self.__inPIPE_Pool = inPIPE
      self.__inNums = len(inPIPE)
    
    else:
      assert isinstance(inPIPE, PIPE)
      assert not inPIPE.is_outlocked(), "The output port of PIPE has already been locked. Please release it firstly."
      password = inPIPE.lock_out()
      assert password is not None
      self.__inPIPE_Pool.append( inPIPE )
      self.__inPassword_Pool.append( password )
      self.__inNums += 1

  def start(self,inPIPE=None):
    '''
    Start running a process to handle Packets in inPIPE.
    '''
    # 1. If this is a silent joint
    if self.coreThread is None:
      if inPIPE is None:
        assert self.__inNums > 0, f"{self.name}: No input PIPEs avaliable."
      else:
        # Link (or redirect) the input PIPE
        self.link( inPIPE )
      # Activate the output PIPE
      for pipe in self.__outPIPE_Pool:
        pipe.activate()
      # Run core process
      for pipe in self.__inPIPE_Pool:
        if pipe.state_is_(mark.silent):
          pipe.activate()
      # Run core process
      self.__coreThread = self._create_thread(self.__core_thread_loop_wrapper)

    # 2. If this is not silent component
    elif self.coreThread.is_alive():
      # 2.1 If this joint is stranded
      if self.__outPIPE_Pool[0].state_is_(mark.stranded):
        ## Check whether it is necessary to redirect
        needRedirect = False
        if inPIPE is not None:
          if isinstance(inPIPE,PIPE):
            inPIPE = [inPIPE,]
          else:
            assert isinstance(inPIPE,(list,tuple))
            inPIPE = list(set(inPIPE))
          
          if len(inPIPE) != self.__inNums:
            needRedirect = True
          else:
            inObjIDs = [ pipe.objid for pipe in self.__inPIPE_Pool ]
            for pipe in inPIPE:
              if pipe.objid not in inObjIDs:
                needRedirect = True
                break
        ## 
        if needRedirect is False:
          for pipe in self.__inPIPE_Pool + self.__outPIPE_Pool:
            pipe.activate()
        ## If need to redirect input PIPE
        else:
          # Close the core process
          self.__redirect_flag = True
          self.wait()
          self.__redirect_flag = False
          # Link the new input PIPE
          self.link( inPIPE )
          # Activate
          for pipe in self.__outPIPE_Pool:
            pipe.activate()
          # Activate
          for pipe in self.__inPIPE_Pool:
            if pipe.state_is_(mark.silent):
              pipe.activate()
          # Run core process
          self.__coreThread = self._create_thread(self.__core_thread_loop_wrapper)

    else:
      raise Exception(f"{self.name}: Can only start a silent or restart a stranded Component.")

  def _create_thread(self,func):
    '''
    Create and start the core thread.
    '''
    coreThread = threading.Thread(target=func)
    coreThread.setDaemon(True)
    coreThread.start()
    return coreThread

  def decide_state(self):

    # Check whether there is silent PIPE
    states = set()
    for pipe in self.__inPIPE_Pool + self.__outPIPE_Pool:
      states.add( pipe.state )
    assert mark.silent not in states, "Can not decide state because input PIPE or outPIPE have not been activated."
    
    # If all PIPEs are the same state
    if len(states) == 1:
      return None, states.pop()
    
    # firstly check whether there is wrong state
    # if there is, terminate all input and output PIPEs instantly
    #  in state might be: active, wrong, terminated, stranded
    # out state might be: active, wrong, terminated, stranded
    if mark.wrong in states:
      for pipe in self.__inPIPE_Pool + self.__outPIPE_Pool:
        if not pipe.state_is_(mark.wrong,mark.terminated):
          pipe.kill()
      return None, mark.wrong
    
    else:
      # collect state flags
      inStates = [ pipe.state for pipe in self.__inPIPE_Pool ]
      outStates = [ pipe.state for pipe in self.__outPIPE_Pool ]
      #  in state might be: active, terminated, stranded
      # out state might be: active, terminated, stranded
      # if output PIPEs has "terminated"  
      if mark.terminated in outStates:
        for pipe in self.__inPIPE_Pool + self.__outPIPE_Pool:
          if not pipe.state_is_(mark.terminated):
            pipe.stop()
        return mark.outPIPE, mark.terminated
      #  in state might be: active, terminated, stranded
      # out state might be: active, stranded
      else:
        # firstly, compare the lastest active flag and stranded flag
        strandedStamps = []
        activeStamps = []
        for pipe in self.__inPIPE_Pool + self.__outPIPE_Pool:
          if pipe.state_is_(mark.stranded):
            strandedStamps.append( pipe.state )
          elif pipe.state_is_(mark.active):
            activeStamps.append( pipe.state )
        # if no stranded flag existed
        if len(strandedStamps) == 0:
          # if terminated in in PIPEs
          if mark.terminated in inStates:
            return mark.inPIPE, mark.terminated
          # if all flags are active
          else:
            return None, mark.active
        # if no active flag existed
        elif len(activeStamps) == 0:
          return None, mark.stranded
        # if active and stranded flag existed at the same time 
        else:
          # if stranded flag is later than active flag
          if max(strandedStamps) > max(activeStamps):
            for pipe in self.__inPIPE_Pool + self.__outPIPE_Pool:
              if pipe.state_is_(mark.active):
                pipe.pause()
            return None, mark.stranded
          # if active flag is later than stranded flag
          else:
            for pipe in self.__inPIPE_Pool + self.__outPIPE_Pool:
              if pipe.state_is_(mark.stranded):
                pipe.activate()
            if mark.terminated in inStates:
              return mark.inPIPE, mark.terminated
            else:
              return None, mark.active

  def __core_thread_loop_wrapper(self):
    self.__core_thread_over = False
    print_(f"{self.name}: Start...")
    try:
      self.core_loop()
    except Exception as e:
      for pipe in self.__inPIPE_Pool + self.__outPIPE_Pool:
        if not pipe.state_is_(mark.wrong,mark.terminated):
          pipe.kill()
      raise e
    else:
      for pipe in self.__inPIPE_Pool + self.__outPIPE_Pool:
        if not pipe.state_is_(mark.wrong,mark.terminated):
          pipe.stop()
    finally:
      print_(f"{self.name}: Stop!")
      self.__core_thread_over = True

  def core_loop(self):

    timecost = 0
    idmaker = None
    buffer = [ None for i in range(self.__inNums) ]

    while True:
      
      ###########################################
      # Decide state
      ############################################

      master,state = self.decide_state()

      ###########################################
      # Decide whether it need to get packet or terminate
      ############################################

      if state == mark.active:
        # If joint is active, skip to picking step
        pass
      elif state == mark.wrong:
        # If joint is wrong, break loop and terminate
        break
      elif state == mark.stranded:
        # If joint is stranded, wait (or terminate)
        time.sleep( info.TIMESCALE )
        if self.__redirect_flag == True:
          break
        continue
      else:
        # if joint is terminated
        ## if outPIPE is terminated, break loop and terminated
        if master == mark.outPIPE:
          break
        ## get packet 
        else:
          ## If packets are exhausted in (at least) one PIPE, stop joint and terminated
          over = False
          for pipe in self.__inPIPE_Pool:
            if pipe.state_is_(mark.terminated) and pipe.is_empty():
              for pipe in self.__inPIPE_Pool + self.__outPIPE_Pool:
                pipe.stop()
              over = True
              break
          if over:
            break
          ## else continue to pick packets
          pass # -> pick packets

      ###########################################
      # Picking step
      ############################################

      # fill input buffer with packets 
      for i, buf in enumerate(buffer):
        if buf is None:
          if self.__inPIPE_Pool[i].is_empty():
            ## skip one time
            continue
          else:
            ## Get a packet
            packet = self.get_packet(i)
            ## Verify the idmaker
            ## Only match packets that their chunk IDs are maked by the same idmaker.
            if not is_endpoint( packet ):
              if idmaker is None:
                idmaker = packet.idmaker
              else:
                assert idmaker == packet.idmaker, "id makers of all input PIPEs do not match."
            ## storage packet
            buffer[i] = packet

      # If buffer has not been filled fully
      if None in buffer:
        time.sleep( info.TIMESCALE )
        timecost += info.TIMESCALE
        ## If timeout, break loop and terminate
        if timecost > info.TIMEOUT:
          print(f"{self.name}: Timeout!")
          for pipe in self.__inPIPE_Pool + self.__outPIPE_Pool:
            pipe.kill()
          break
        ## try to fill again
        else:
          continue

      ## If buffer has been filled fully
      else:
        #### Match the chunk id
        cids = [ x.cid for x in buffer ]
        maxcid = max( cids )
        for i,pack in enumerate(buffer):
          if pack.cid != maxcid:
            buffer[i] = None
        ##### If chunk ids does not match, only keep the latest packets
        ##### Remove mismatch packets and try fill again
        if None in buffer:
          continue
        ##### If chunk ids matched
        else:
          ### If all packets are empty (Especially when they are the endpoint, the possibility is very high).
          numsEndpoint = sum( [ int(is_endpoint(pack)) for pack in buffer ] )
          assert numsEndpoint == 0 or numsEndpoint == self.__inNums
          numsEmpty = sum( [ int(pack.is_empty()) for pack in buffer ] )
          if numsEmpty == self.__inNums:
            if is_endpoint(buffer[0]):
              for i in range( self.__outNums ):
                self.put_packet( i, Endpoint(cid=maxcid,idmaker=idmaker) )
            else:
              for i in range( self.__outNums ):
                self.put_packet( i, Packet(items={},cid=maxcid,idmaker=idmaker) )
          else:
            ###### Do joint operation according to specified rules.
            inputs = [ dict(pack.items()) for pack in buffer ]
            outputs = self.__joint_function( inputs )
            ###### Verify results
            if isinstance(outputs,dict):
              outputs = [ outputs, ]
            else:
              assert isinstance(outputs,(tuple,list))
              for output in outputs:
                assert isinstance(output,dict)
            assert len(outputs) == self.__outNums
            ###### Append results into output PIPEs
            if is_endpoint(buffer[0]):
              for i in range(self.__outNums):
                self.put_packet( i, Endpoint( items=outputs[i], cid=maxcid, idmaker=idmaker) )
            else:
              for i in range(self.__outNums):
                self.put_packet( i, Packet( items=outputs[i], cid=maxcid, idmaker=idmaker) )
            ###### clear buffer and fill again
            for i in range(self.__inNums):
              buffer[i] = None
            
            continue

  def stop(self):
    '''
    Terminate this component normally.
    Note that we do not terminate the core process by this function.
    We hope the core process can be terminated with a mild way.
    '''
    # Stop input PIPE
    for pipe in self.__inPIPE_Pool:
      pipe.stop()
  
  def kill(self):
    '''
    Terminate this component with state: wrong.
    It means errors occurred somewhere.
    Note that we do not kill the core thread by this function.
    We hope the core thread can be terminated with a mild way.
    '''
    # Kill input PIPE
    for pipe in self.__inPIPE_Pool:
      pipe.kill()
  
  def pause(self):
    '''
    Pause the Componnent
    '''
    # Kill input PIPE
    for pipe in self.__inPIPE_Pool:
      pipe.pause()

  def wait(self):
    '''
    Wait until the core thread is finished.
    '''
    if self.__coreThread is None:
      raise Exception(f"{self.name}: Component has not been started.")
    else:
      self.__coreThread.join()
      #while not self.__core_thread_over:
      #  time.sleep(info.TIMESCALE)
      #self.__coreThread.terminate()

  def get_packet(self,inID):
    '''
    Get packet from input PIPE.
    '''
    assert len(self.__inPIPE_Pool) > 0
    return self.__inPIPE_Pool[inID].get(password=self.__inPassword_Pool[inID])
  
  def put_packet(self,outID,packet):
    self.__outPIPE_Pool[outID].put(packet,password=self.__outPassword_Pool[outID])

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
        #temp = " ".join( [ str(v) for v in value[:10] ] )
        #out.append( f"{key}: [ {temp} ...] " )
        out.append( f"{key}: {value} " )
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
      if not packet.is_empty():
        print_()
        mapFunc( packet )
      print_(f"----- Endpoint -----")
      continue
    else:
      print_()
      mapFunc( packet )

  lastState = "terminated" if pipe.state_is_(mark.terminated) else "wrong"
  print_( f"Final state of this PIPE: {lastState} \n Time report: {pipe.report_time()}" )

def dynamic_run(target,inPIPE=None,items=["data"]):
  print_("exkaldirt.base.dynamic_run has been removed from version 1.2.0. See exkaldirt.base.dynamic_display function.")

class ContextManager(ExKaldiRTBase):
  '''
  Context manager.
  '''
  def __init__(self,left,right,name=None):
    super().__init__(name=name)
    assert isinstance(left,int) and left >= 0
    assert isinstance(right,int) and right >= 0
    self.__left = left
    self.__right = right
    self.__buffer = None

  @property
  def left(self):
    return self.__left
  
  @property
  def right(self):
    return self.__right

  def __compute_size(self,center):
    assert center >= self.__left and center >= self.__right
    self.__center = center
    if self.__right > 0:
      self.__width = self.__left + center + center
    else:
      self.__width = self.__left + center
    self.__tail = self.__left + center + self.__right

  def wrap(self, batch):
    '''
    Storage a batch frames (matrix) and return the new frames wrapped with left and right context.
    If right context > 0, we will storage this batch data and return the previous batch data, 
    and None will be returned at the first step.

    Args:
      _batch_: (2d numpy array).
    
    Return:
      (None, 2d numpy array).
    '''
    assert isinstance(batch,np.ndarray) and len(batch.shape) == 2
    assert 0 not in batch.shape
    if self.__buffer is None:
      frames, dim = batch.shape
      self.__compute_size(frames)
      self.__buffer = np.zeros([self.__width,dim],dtype=batch.dtype)
      if self.__right == 0:
        self.__buffer[self.__left:,:] = batch
        return self.__buffer.copy()
      else:
        self.__buffer[-self.__center:,:] = batch
        return None
    else:
      assert len(batch) == self.__center
      if self.__right == 0:
        self.__buffer[0:self.__left,:] = self.__buffer[ self.__center: ]
        self.__buffer[self.__left:,:] = batch
        return self.__buffer.copy()
      else:
        self.__buffer[ 0:-self.__center,:] = self.__buffer[ self.__center:,: ]
        self.__buffer[ -self.__center:,:] = batch
        return self.__buffer[0:self.__tail,:].copy()

  def strip(self,batch):
    '''
    Strip the context.

    Args:
      _batch_: (2d numpy array).
    
    Return:
      (2d numpy array).
    '''
    assert isinstance(batch,np.ndarray) and len(batch.shape) == 2
    assert batch.shape[0] == self.__tail
    return batch[ self.__left: self.__left + self.__center ]
