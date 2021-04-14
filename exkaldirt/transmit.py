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
from exkaldirt.base import info, mark, ENDPOINT, is_endpoint, NullPIPE
from exkaldirt.utils import uint_to_bytes, uint_from_bytes

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

ActiveMark = b"0"
EndpointMark = b"1"
TerminatedMark = b"2"
ErrorMark = b"3"
StrandedMark = b"4"

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
    bsize = uint_to_bytes(size,length=4) # bsize: 4
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
      size1 = uint_from_bytes( bsize1 )
      size2 = uint_from_bytes( bsize2 )
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
  def __init__(self,thost,tport,name=None):
    '''
    Args:
      _thost_: (str) Target host IP.
      _tport_ : (int) Target port ID.
      _batchSize_: (int) Batch size.
      _name_: (str) Name.
    '''
    super().__init__(oKey="null",name=name)
    self.__rAddr = (thost,tport)


  def core_loop(self):

    self.__proto = SendProtocol(thost=self.__rAddr[0],
                                tport=self.__rAddr[1],
                                name=self.basename+" Send Protocol"
                              )
    
    try:
      while True:

        action = self.decide_action()

        if action is False:
          self.__proto.send( ErrorMark )
          break
        elif action is None:
          self.__proto.send( TerminatedMark )
          continue
        else:
          packet = self.get_packet()
          if is_endpoint(packet):
            self.__proto.send( EndpointMark )
          else:
            message = packet.encode()
            self.__proto.send( ActiveMark + message )
    
    finally:
      self.__proto.close()

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
    super().__init__(oKey="null",name=name)
    # Define the protocol
    self.__bport = bport
    

  def core_loop(self):

    self.__proto = ReceiveProtocol(bport=self.__bport)

    try:
      while True:
        master, state = self.decide_state()
        if state == mark.wrong:
          break
        elif mark




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

  def link(self,inPIPE=None,iKey=None):
    if inPIPE is None:
      inPIPE = NullPIPE()
    super().link( inPIPE=inPIPE )

  def start(self,inPIPE=None,iKey=None):
    if inPIPE is None:
      inPIPE = NullPIPE()
    super().start( inPIPE=inPIPE )