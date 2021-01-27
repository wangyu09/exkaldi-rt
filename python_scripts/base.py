# coding=utf-8
#
# Yu Wang (University of Yamanashi)
# Jan, 2021
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

TIMEOUT = 10 #Seconds 
TIMESCALE = 0.01 #Seconds
CMDROOT = "."

def get_floot_floor():
  '''Get the floot floor value'''
  cmd = os.path.join(CMDROOT,"get-float-floor")
  p = subprocess.Popen(cmd,stdout=subprocess.PIPE)
  (out,_) = p.communicate()
  return float(out.decode().strip())

EPSILON = get_floot_floor()

'''A base class to descript state'''
class StateFlag:
  
  ALIVE = 0
  TERMINATION = 1
  ERROR = 2

  def __init__(self):
    self.__state = StateFlag.ALIVE
  
  def is_alive(self):
    return self.__state == StateFlag.ALIVE
  
  def is_error(self):
    return self.__state == StateFlag.ERROR
  
  def is_termination(self):
    return self.__state == StateFlag.TERMINATION
  
  def shift_state_to_error(self):
    self.__state = StateFlag.ERROR

  def shift_state_to_termination(self):
    self.__state = StateFlag.TERMINATION

'''A object to connect different components to pass data packets'''
class PIPE(StateFlag):
  '''
  It is a Last-In-Last-Out queue.
  '''
  def __init__(self):
    super().__init__()
    self.__cache = queue.Queue()
    self.__extra_info = None
  
  def set_error(self):
    self.shift_state_to_error()
    self.__cache.queue.clear()
    self.__extra_info = None
  
  def set_termination(self):
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
    Can only get new item from ALIVE and TERMINATION PIPE.
    '''
    if self.is_termination() and self.is_empty():
      raise Exception("PIPE is terminated and nothing is left.") 
    elif self.is_error():
      raise Exception("Can not get item from a killed PIPE." )
    return self.__cache.get(timeout=TIMEOUT)
  
  def put(self,packet):
    '''
    Can only append new packet into ALIVE PIPE.
    '''
    if not self.is_alive():
      raise Exception("Can only append item into ALIVE PIPE." )
    assert isinstance(packet,Packet), "This is not a Packet object."
    self.__cache.put(packet)
  
  def add_extra_info(self,info=None):
    self.__extra_info = info
  
  def get_extra_info(self):
    return self.__extra_info

  def to_list(self,deep=True):
    '''Convert PIPE to list.'''
    assert isinstance(deep,bool), "<deep> need a bool value."
    size = self.size()
    if deep:
      return [ (self.__cache.get()).item for i in range(size) ]
    else:
      return [ self.__cache.get() for i in range(size) ]

'''Data classes for different stream data passed through PIPE'''

class Packet:
  def __init__(self,item,endpoint=False):
    assert isinstance(endpoint,bool), "<endpoint> need a bool value."
    self.__item = item
    self.__endpoint = endpoint
  @property
  def item(self):
    return self.__item
  def is_endpoint(self):
    return self.__endpoint

class Element(Packet):
  def __init__(self,item,endpoint=False):
    if isinstance(item,int):
      item = np.int16(item)
    elif isinstance(item,float):
      item = np.int32(item)
    else:
      assert isinstance(item,(np.int8,np.int16,np.int32,
                             np.float16,np.float32,np.float64)), "Element packet must be int or float value."
    super().__init__(item,endpoint)
  
  @property
  def dtype(self):
    return str(self.item.dtype)
  
class Vector(Packet):
  def __init__(self,item,endpoint=False):
    assert isinstance(item,np.ndarray) and len(item.shape) == 1, "Vector packet must be 1-d numpy array."
    super().__init__(item,endpoint) 

  @property
  def dtype(self):
    return str(self.item.dtype)
  
class BVector(Packet):
  def __init__(self,item,dtype,endpoint=False):
    assert isinstance(item,bytes), "BVector packet must be bytes object."
    assert dtype in ["int16","float32"]
    super().__init__(item,endpoint)
    self.__dtype = dtype
  
  @property
  def dtype(self):
    return self.__dtype
  
  def decode(self):
    return Vector(np.frombuffer(self.item,dtype=self.__dtype),endpoint=self.is_endpoint())

class Best1(Packet):
  def __init__(self,item,endpoint=False):
    assert isinstance(item,str), "Best1 packet must be string."
    super().__init__(item,endpoint)
  
  @property
  def dtype(self):
    return "str"

'''Other tools'''
# the object to pass intermediate data

def encode_vector(vec):
  return (" " + " ".join( map(str,vec)) + " ").encode()

def run_shell_command(cmd,inputs=None):

  if inputs is not None:
    assert isinstance(inputs,bytes)
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
