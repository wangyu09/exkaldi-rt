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

from exkaldi2.base import ExKaldi2Base,Component,PIPE,Vector,Element,Text
from exkaldi2.base import info

socket.setdefaulttimeout(info.TIMEOUT)

def get_host_ip():
  '''Get the IP address of local host.'''
  try:
    testclient = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    testclient.connect(('8.8.8.8',80))
    ip = testclient.getsockname()[0]
  finally:
    testclient.close()
  
  return ip

# This is a example function to encode value packet in order to send it through network.
def encode_value_packet(packets):
  assert isinstance(packets,list), '<packets> must be a list of packets.'
  # get the information
  frames = len(packets)
  assert frames > 0, 'Packets is empty.'
  # prepare endpoint flag
  flags = np.zeros([frames,],dtype=np.uint8)
  datas = []
  for i in range(frames):
    flags[i] = int(packets[i].is_endpoint())
    datas.append( packets[i].data )

  # prepare dtype
  dtype = packets[0].dtype
  if dtype.startswith("int"):
    formatFlag = b"I"
    formatFlag += np.int8(dtype[3:]).tobytes()
  elif dtype.startswith("float"):
    formatFlag = b"F"
    formatFlag += np.int8(dtype[5:]).tobytes()
  else:
    raise Exception(f"Unknown packet dtype: {dtype}")
  if isinstance(packets[0],BVector):
    datas = b"".join(datas)
  else:
    datas = np.array(datas).tobytes()
  
  # encoding
  result = np.int32(frames).tobytes()
  result += formatFlag
  result += flags.tobytes()
  result += datas

  return result 

# This is a example function to decode value packet received from network.
def decode_value_packet(pointer):

  results = []
  # read frames
  frames = int(np.frombuffer(pointer.read(4),dtype=np.uint32)[0])
  # read dtype
  formatFlag = pointer.read(1)
  if formatFlag == b"I":
    formatFlag = "int"
  elif formatFlag == b"F":
    formatFlag = "float"
  else:
    raise Exception(f"Unknown flag: {formatFlag}")
  
  width = int(np.frombuffer(pointer.read(1),dtype=np.uint8)[0])
  formatFlag += str(width)
  # read flags
  flags = np.frombuffer(pointer.read(frames),dtype=np.uint8)
  # read contexts
  body = pointer.read()
  data = np.frombuffer(body,dtype=formatFlag)
  if len(data) == frames:
    for i in range(frames):
      results.append( Element(data[i],endpoint=bool(flags[i])) )
  else:
    data = data.reshape((frames,-1))
    for i in range(frames):
      results.append( Vector(data[i],endpoint=bool(flags[i])) )

  return results

def encode_text_packet(text):
  flag = np.uint8(text.is_endpoint()).tobytes()
  content = (text.data).encode()
  return flag + content

def decode_text_packet(pointer):
  flag = bool(np.frombuffer(pointer.read(1),dtype=np.uint8)[0])
  res = pointer.read().decode().strip()
  return Text(res,endpoint=flag)

class SendProtocol(ExKaldi2Base):

  def __init__(self,thost,tport=9509,name=None):
    super().__init__(name=name)
    self.__client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    print(f"{self.name}: Target Address is ({thost},{tport}). Connect...")
    self.__client.connect((thost,tport),)
    print(f"{self.name}: Connected!")
    self.__raddr = (thost,tport)
  
  def send(self,message):
    assert isinstance(message,bytes)
    # 4 bytes to mark message size
    bsize = np.uint32(len(message)).tobytes()
    counter = 0
    while True:
      self.__client.sendall( bsize + message )
      respon = self.__client.recv(1)
      if respon == b'0':
        break
      elif respon == b'1':
        counter += 1
        if counter >= info.SOCKET_RETRY:
          raise Exception(f"{self.name}: Failed to send the message!")
      else:
        raise Exception(f"{self.name}: Unknown flag: {respon}!")

  def close(self):
    self.__client.close()

  def get_remote_addr(self):
    return self.__raddr

class ReceiveProtocol(ExKaldi2Base):
  
  def __init__(self,bport=9509,name=None):
    super().__init__(name=name)
    bhost = get_host_ip()
    self.__server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    self.__server.bind((bhost,bport),)
    print(f"{self.name}: Host address is ({bhost},{bport}). Listening ...")
    self.__server.listen(1)
    self.__client, self.__raddr = self.__server.accept()
    print(f"{self.name}: Connected! Remote address is ({self.__raddr[0]}, {self.__raddr[1]}).")
    self.__baddr = (bhost,bport)

  def get_host_addr(self):
    return self.__baddr

  def receive(self):
    while True:
      # receive header
      buffer = self.__client.recv(4)  
      bsize = int(np.frombuffer(buffer,dtype=np.uint32)[0])
      # receive body
      buffer = self.__client.recv(bsize)
      # check
      if len(buffer) == bsize:
        self.__client.sendall(b"0")
        return buffer
      else:
        self.__client.sendall(b"1") # require sending again
  
  def get_remote_addr(self):
    return self.__taddr

  def close(self):
    self.__client.close()
    self.__server

class ElementSender(Component):
  
  def __init__(self,thost,tport,batchSize,name=None):
    super().__init__(name=name)
    assert isinstance(batchSize,int) and batchSize > 0, f"{self.name}: <batchSize> must be a positive int value."
    self.__proto = SendProtocol(thost, tport, name=self.name+" Send Protocol")
    self.__batchSize = batchSize
    self.__inputBuffer = []
    self.encode_function = None
  
  def __prepare_chunk_packet(self,elePIPE):
    timeCost = 0
    pos = 0
    while pos < self.__batchSize:
      if self.is_error() or inPIPE.is_error():
        self.__set_error()
        return False
      elif self.is_termination() or inPIPE.is_exhaustion():
        break
      elif inPIPE.is_empty():
        time.sleep( info.TIMESCALE )
        timeCost += info.TIMESCALE
        if timeCost > info.TIMEOUT:
          inPIPE.set_error()
          self.__set_error()
          return False
      else:
        pac = inPIPE.get()
        assert isinstance(pac,(Element,Vector,BVector)), f"TCP Client sending needs Element, Vector, BVector objects but got: {type(pac).__name__}"
        if pos != 0:
          self.__inputBuffer.append(pac)
          pos += 1
        else:
          if pac.is_endpoint():
            continue # discard this packet
          self.__inputBuffer.append(pac)
          pos += 1
    if pos == 0:
      self.__set_termination()
      return False
    return True
  
  def __send(self,inPIPE):
    print("Start sending packets...")
    timecost = 0
    try:
      while True:
        # prepare chunk packet
        self.__inputBuffer.clear()
        if not self.__prepare_chunk_packet(inPIPE):
          break
        # encoding
        message = self.encode_function(self.__inputBuffer)
        assert isinstance(message,bytes), f"Encoding result must be a bytes but got: {type(message).__name__}"
        # send
        self.__proto.send( b"0" + message )
        # if no more data
        if inPIPE.is_exhaustion():
          self.__set_termination()
          break
    except Exception as e:
      inPIPE.set_error()
      self.__set_error()
      raise e
    finally:
      print("Stop sending packets!")
  
  def __set_error(self):
    self.shift_state_to_error()
    self.__proto.send(b"1") # flag(1->error)

  def __set_termination(self):
    self.__proto.send(b"2") # flag(2->error)

  def start(self,inPIPE:PIPE):
    # send the wave information
    extraInfo = inPIPE.get_extra_info()
    assert extraInfo is not None, "Miss Audio Info in inpute PIPE."
    firstMess = np.array([extraInfo.rate,extraInfo.channels,extraInfo.width],dtype=np.uint32).tobytes()
    self.__proto.send(firstMess)

    assert self.encode_function is not None, 'Please implement this function.' 
  
    # Start sending thread
    self.__sendThread = threading.Thread(target=self.__send,args=(inPIPE,))
    self.__sendThread.setDaemon(True)
    self.__sendThread.start()

class PacketReceiver(Component):

  def __init__(self,bport=9509):
    super().__init__()
    self.__proto = ReceiveProtocol(bport=bport)
    self.__outPIPE = PIPE()
    self.__receiveThread = None
    self.decode_function = None
  
  def get_out_pipe(self):
    return self.__outPIPE
  
  def __receive(self):
    print("Start receiving packets...")
    try:
      while True:
        if self.is_error() or self.__outPIPE.is_error():
          self.__set_error()
          break
        elif self.is_termination() or self.__outPIPE.is_termination():
          self.__set_termination()
          break
        else:
          message = self.__proto.receive() #
          with BytesIO(message) as pointer:
            flag = pointer.read(1) 
            if flag == b'0': 
              decodeResults = self.decode_function(pointer)
              for pac in decodeResults:
                self.__outPIPE.put(pac)
            elif flag == b'1': 
              print('Error occurred in remote machine.')
              self.__set_error()
              break
            elif flag == b'2': 
              self.__set_termination()
              break
            else:
              raise Exception(f"Unkown flag: {flag}")

    except Exception as e:
      self.__set_error()
      raise e
    finally:
      print("Stop receiving packet!")

  def __set_error(self):
    '''
    When received error flag.
    '''
    self.shift_state_to_error()
    self.__outPIPE.set_error()
    self.__proto.close()
  
  def __set_termination(self):
    '''
    When received end flag
    '''
    self.shift_state_to_termination()
    self.__outPIPE.set_termination()
    self.__proto.close()
  
  def start_receiving(self):
    '''
    Receive packet from client.
    '''
    assert self.decode_function is not None, 'Please implement this function.' 
    # Receive the audio infomation
    vert = self.__proto.receive() # uint32 * 4
    vert = np.frombuffer(vert,dtype=np.uint32)
    audioInfo = namedtuple("AudioInfo",["rate","channels","width"])(
                                        vert[0], vert[1], vert[2])
    self.__outPIPE.add_extra_info(audioInfo)
    # Receive the data
    self.__receiveThread = threading.Thread(target=self.__receive)
    self.__receiveThread.setDaemon(True)
    self.__receiveThread.start()

  def wait(self):
    if self.__receiveThread:
      self.__receiveThread.join()
