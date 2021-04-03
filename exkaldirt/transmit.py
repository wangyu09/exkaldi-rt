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

import time
import socket
import threading
from collections import namedtuple
import numpy as np
from io import BytesIO

from exkaldirt.base import ExKaldiRTBase, Component, PIPE, Vector, Element, Text
from exkaldirt.base import info, ENDPOINT, is_endpoint

socket.setdefaulttimeout(info.TIMEOUT)

def get_host_ip():
  '''
  Get the IP address of local host.
  '''
  try:
    testclient = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    testclient.connect(('8.8.8.8',80))
    ip = testclient.getsockname()[0]
  finally:
    testclient.close()
  
  return ip

def encode_size(size):
  assert isinstance(size,int)
  return size.to_bytes(length=4,byteorder="little",signed=False)

def decode_size(bsize):
  assert isinstance(bsize,bytes)
  return int.from_bytes(bsize,byteorder="little",signed=False)

def encode_dtype(dtype):
  assert isinstance(dtype,str)
  if dtype.startswith("int"):
    formatFlag = b"I"
    formatFlag += int(dtype[3:]).to_bytes(length=1,byteorder="little",signed=False)
  elif dtype.startswith("float"):
    formatFlag = b"F"
    formatFlag += int(dtype[5:]).to_bytes(length=1,byteorder="little",signed=False)
  else:
    raise Exception(f"Unknown packet dtype: {dtype}")
  return formatFlag

def decode_dtype(bdtype):
  assert isinstance(bdtype,bytes) and len(bdtype) == 2
  formatFlag = bdtype[0:1]
  if formatFlag == b"I":
    formatFlag = "int"
  elif formatFlag == b"F":
    formatFlag = "float"
  else:
    raise Exception(f"Unknown flag: {formatFlag}")
  
  width = int.from_bytes(bdtype[1:],byteorder="little",signed=False)
  formatFlag += str(width)

  return formatFlag

# This is a example function to encode value packet in order to send it through network.
def encode_value_packet(packets):
  '''
  Encode Element or Vector packets.
  '''
  assert isinstance(packets,(list,tuple))
  if len(packets) == 0:
    return b""
  # frames and dim. and data
  frames = len(packets)
  chunk = []
  for pac in packets:
    chunk.append( pac.data )
  chunk = np.array( chunk )
  if len(chunk.shape) == 1:
    dim = 1
  else:
    assert len(chunk.shape) == 2
    dim = chunk.shape[1]
  datas = chunk.tobytes()

  # encode
  result = encode_size(frames)
  result += encode_size(dim)
  result += encode_dtype(packets[0].dtype)
  result += datas

  # results = frames[4] + dim[4] + "F"/"I"[1] + typesize[1] + datas
  return result 

# This is a example function to decode value packet received from network.
def decode_value_packet(items):

  results = []
  # read frames
  frames = decode_size( items[0:4] ) 
  dim = decode_size( items[4:8] ) 
  # read dtype
  dtype = decode_dtype( items[8:10] )
  # read contexts
  data = np.frombuffer( items[10:], dtype=dtype )
  if dim == 1:
    for i in range(frames):
      results.append( Element(data[i]) )
  else:
    data = data.reshape( (frames,dim) )
    for i in range(frames):
      results.append( Vector(data[i]) )

  return results

def encode_text_packet(packets):
  '''
  Encode Text packets.
  '''
  assert isinstance(packets,(list,tuple))
  if len(packets) == 0:
    return b""
  # frames and dim. and data
  chunk = []
  for pac in packets:
    chunk.append( pac.data )
  chunk = " # ".join( chunk )
  chunk = chunk.encode()

  return chunk

def decode_text_packet(items):
  '''
  Decode Text packets.
  '''
  results = []
  items = items.decode()
  if len(items) == 0:
    return []
  items = items.split("#")
  for item in items:
    results.append( Text(item) )
  return results

ActiveMark = b"0"
EndpointMark = b"1"
TerminatedMark = b"2"
ErrorMark = b"3"

class SendProtocol(ExKaldiRTBase):
  '''
  Packet Sending Protocol.
  '''
  def __init__(self,thost,tport=9509,name=None):
    super().__init__(name=name)
    # Open a client
    self.__client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    # Connect to remote host.
    self.__connect_to(thost,tport)
    #
    self.__raddr = (thost,tport)

  def __connect_to(self,thost,tport):
    '''
    Connect to remote host.
    '''
    print(f"{self.name}: Target Address is ({thost},{tport}). Connecting...")
    timecost = 0
    while True:
      try:
        self.__client.connect((thost,tport),)
      except (ConnectionRefusedError,ConnectionAbortedError):
        time.sleep(info.TIMESCALE)
        timecost += info.TIMESCALE
        if timecost >= info.TIMEOUT:
          raise Exception(f"{self.name}: Timeout! No remote host is activated!")
        continue
      else:
        break
    print(f"{self.name}: Connected!")

  def send(self,message):
    '''
    Send a message.

    Args:
      _message_: (bytes) a bytes object.
    '''
    assert isinstance(message,bytes), "_message_ should be a bytes object."
    # 4 bytes to mark message size
    size = len(message)
    assert size > 0, f"{self.name}: Can not send an empty packet."
    bsize = encode_size(size) # bsize: 4
    # 
    retryCounter = 0
    while True:
      ## Send this message
      ## We will add two size flags to verify this message
      self.__client.sendall( bsize + bsize + message )
      ## Listen the response
      respon = self.__client.recv(1)
      
      if respon == b'0':
        break
      elif respon == b'1':
        retryCounter += 1
        if retryCounter >= info.SOCKET_RETRY:
          raise Exception(f"{self.name}: Failed to send the message!")
        continue
      else:
        raise Exception(f"{self.name}: Unknown remote transmition response: {respon}!")

  def close(self):
    self.__client.close()

  def get_remote_addr(self):
    return self.__raddr

class ReceiveProtocol(ExKaldiRTBase):
  
  def __init__(self,bport=9509,name=None):
    super().__init__(name=name)
    # Get the local IP address
    bhost = get_host_ip()
    # Bind local host
    self.__server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    self.__server.bind((bhost,bport),)
    # Connect the remote host
    self.__connect_from(bhost,bport)
    #
    self.__baddr = (bhost,bport)
    #

  def __connect_from(self,bhost,bport):
    '''
    Connect from remote host.
    '''
    print(f"{self.name}: Host address is ({bhost},{bport}). Listening ...")
    self.__server.listen(1)
    self.__client, self.__raddr = self.__server.accept()
    print(f"{self.name}: Connected! Remote address is ({self.__raddr[0]}, {self.__raddr[1]}).")

  def get_host_addr(self):
    return self.__baddr

  def receive(self):
    '''
    Receive passage.
    '''
    while True:
      # 1 verify byte size
      bsize1 = self.__client.recv(4)
      bsize2 = self.__client.recv(4)
      size1 = decode_size( bsize1 )
      size2 = decode_size( bsize2 )
      if size1 != size2:
        ## If size does not match
        ## 1 trye to clear receiving buffer
        self.__client.recv( info.MAX_SOCKET_BUFFER_SIZE )
        ## 2 require sending again
        self.__client.sendall( b"1" )
        continue
      # 2 verify data
      buffer = self.__client.recv( size1 )
      # check
      if len(buffer) == size1:
        # Tell the remote host "received successfully"
        self.__client.sendall( b"0" )
        return buffer
      else:
        ## 1 trye to clear receiving buffer
        self.__client.recv(info.MAX_SOCKET_BUFFER_SIZE)
        ## 2 require sending again
        self.__client.sendall(b"1")

  def get_remote_addr(self):
    return self.__raddr

  def close(self):
    self.__client.close()
    self.__server.close()

class PacketSender(Component):
  '''
  Send packets using a remote connection.
  The outPIPE of PacketSender is just a dummy output PIPE.
  '''
  def __init__(self,thost,tport,batchSize,name=None):
    '''
    Args:
      _thost_: (str) Target host IP.
      _tport_ : (int) Target port ID.
      _batchSize_: (int) Batch size.
      _name_: (str) Name.
    '''
    super().__init__(name=name)
    assert isinstance(batchSize,int) and batchSize > 0, f"{self.name}: <batchSize> must be a positive int value."
    self.__proto = SendProtocol(thost, tport, name=self.name+" Send Protocol")
    self.__batchSize = batchSize
    self.__elementBuffer = []
    self.encode_function = None
    self.__reset_position_flag()
    
    self.__debug_counter = 0

  @property
  def count(self):
    self.__debug_counter += 1
    return self.__debug_counter - 1

  def __reset_position_flag(self):
    self.__endpointStep = False
    self.__finalStep = False

  def __prepare_chunk_packet(self,inPIPE):
    timeCost = 0
    pos = 0
    while pos < self.__batchSize:
      if inPIPE.is_wrong():
        self.kill()
        return False
      elif inPIPE.is_exhausted():
        self.__finalStep = True
        break
      elif inPIPE.is_empty():
        time.sleep( info.TIMESCALE )
        timeCost += info.TIMESCALE
        if timeCost > info.TIMEOUT:
          print(f"{self.name}: Timeout! Did not receive any data for a long time！")
          inPIPE.kill()
          self.kill()
          return False
      # If need wait because of blocked
      elif inPIPE.is_blocked():
        time.sleep(info.TIMESCALE)
      else:
        pac = inPIPE.get()
        if is_endpoint(pac):
          self.__endpointStep = True
          break
        else:
          self.__elementBuffer.append(pac)
          pos += 1
    return True
  
  def __send(self,inPIPE):
    print(f"{self.name}: Start...")
    timecost = 0
    try:
      while True:
        # prepare chunk packet
        self.__elementBuffer.clear()
        if not self.__prepare_chunk_packet(inPIPE):
          break
        # encoding
        if len(self.__elementBuffer) > 0:
          message = self.encode_function(self.__elementBuffer)
        assert isinstance(message,bytes), f"Encoding result must be a bytes but got: {type(message).__name__}"
        if self.is_wrong():
          inPIPE.kill()
          self.kill()
          break
        else:
          # send
          if len(self.__elementBuffer) > 0:
            self.__proto.send( ActiveMark + message )
          ## If arrived ENDPOINT
          if self.__endpointStep:
            self.__proto.send( EndpointMark )
            self.__reset_position_flag()
          ## If over
          if self.__finalStep or self.is_terminated():
            self.stop()
            break
    except Exception as e:
      inPIPE.kill()
      self.kill()
      raise e
    finally:
      print(f"{self.name}: Stop!")
  
  def _start(self,inPIPE:PIPE):
    # send the wave information
    assert self.encode_function is not None, 'Please implement this function.' 
  
    # Start sending thread
    sendThread = threading.Thread(target=self.__send,args=(inPIPE,))
    sendThread.setDaemon(True)
    sendThread.start()

    return sendThread

  def kill(self):
    super().kill()
    self.__proto.send( ErrorMark )

  def stop(self):
    super().stop()
    self.__proto.send( TerminatedMark )

class PacketReceiver(Component):
  '''
  Receive packets using a remote connection.
  '''
  def __init__(self,bport=9509,name=None):
    '''
    Args:
      _bport_: (int) Bind port ID.
      _name_: (str) name.
    '''
    super().__init__(name=name)
    # Define the protocol
    self.__proto = ReceiveProtocol(bport=bport)
    self.decode_function = None

  def __receive(self):
    print(f"{self.name}: Start...")
    try:
      while True:
        if self.is_wrong() or \
           self.outPIPE.is_wrong() or \
           self.outPIPE.is_terminated():
          self.kill()
          break
        else:
          # Receive a message
          message = self.__proto.receive()
          # Check the flag
          flag = message[0:1]
          # If this is active packet
          if flag == ActiveMark: 
            decodeResults = self.decode_function( message[1:] )
            for pac in decodeResults:
              self.outPIPE.put( pac )
          # If this is an endpoint flag
          elif flag == EndpointMark:
            self.outPIPE.put( ENDPOINT )
          # If this is an error flag
          elif flag == ErrorMark: 
            print(f'{self.name}: Error occurred in remote machine.')
            self.kill()
            break
          # If this is a termination flag
          elif flag == TerminatedMark: 
            self.stop()
            break
          else:
            raise Exception(f"{self.name}: Unkown flag {flag}")

    except Exception as e:
      self.kill()
      raise e

    finally:
      print(f"{self.name}: Stop!")

  def kill(self):
    '''
    When received error flag.
    '''
    super().kill()
    self.__proto.close()

  def stop(self):
    '''
    When received end flag
    '''
    super().stop()
    self.__proto.close()
  
  def _start(self,inPIPE:PIPE):
    '''
    Receive packet from client.
    '''
    assert self.decode_function is not None, 'Please implement this function.' 
    
    # Receive the data
    receiveThread = threading.Thread(target=self.__receive)
    receiveThread.setDaemon(True)
    receiveThread.start()

    return receiveThread

  def start(self,inPIPE=None):
    '''_inPIPE_ is just a place holder'''
    super().start(None)