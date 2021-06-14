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

from exkaldirt.base import ExKaldiRTBase, Component, PIPE, Packet
from exkaldirt.base import info, mark, Endpoint, is_endpoint, NullPIPE
from exkaldirt.utils import * 

# from base import ExKaldiRTBase, Component, PIPE, Packet
# from base import info, mark, Endpoint, is_endpoint, NullPIPE
# from utils import *

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
PacketMark = b"5"

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
      try:
        self.__client.sendall( bsize + bsize + message )
      except Exception as e:
        return ErrorMark
        ## Listen the response
      respon = self.__client.recv(1)
      
      if respon == b'0':
        fblen = uint_from_bytes( self.__client.recv(1) )
        return None if fblen == 0 else self.__client.recv(fblen)
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

  def receive(self,feedback=None):
    '''
    Receive passage.
    '''
    if feedback is None:
      feedback = b""
    else:
      assert isinstance(feedback,bytes) and len(feedback) <= 256

    while True:
      # 1 verify byte size
      bsize1 = self.__client.recv(4)
      bsize2 = self.__client.recv(4)
      size1 = uint_from_bytes( bsize1 )
      size2 = uint_from_bytes( bsize2 )
      if size1 != size2:
        ## If size does not match
        ## 1 try to clear receiving buffer
        self.__client.recv( info.MAX_SOCKET_BUFFER_SIZE )
        ## 2 require sending again
        self.__client.sendall( b"1" )
        continue
      # 2 verify data
      buffer = self.__client.recv( size1 )
      # check
      if len(buffer) == size1:
        # Tell the remote host "received successfully"
        self.__client.sendall( b"0" + uint_to_bytes(len(feedback),length=1) + feedback )
        return buffer
      else:
        ## 1 trye to clear receiving buffer
        self.__client.recv(info.MAX_SOCKET_BUFFER_SIZE)
        ## 2 require sending again
        self.__client.sendall(b"1" )

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
    timecost = 0
    try:
      while True:

        ###############################
        # Decide the state and action
        # The outPIPE 
        ###############################

        if self.inPIPE.state_is_(mark.wrong):
          if not self.outPIPE.state_is_(mark.wrong,mark.terminated):
            self.outPIPE.kill()
          # No matter what the state of the remote, kill local
          _ = self.__proto.send( ErrorMark )
          break

        elif self.inPIPE.state_is_(mark.stranded):
          # Tell remote host the state and get the feedback
          feedback = self.__proto.send( StrandedMark + double_to_bytes(self.inPIPE.timestamp) )
          # Check the feedback information
          # If remote state is wrong
          if feedback[0:1] == ErrorMark:
            if not self.inPIPE.state_is_(mark.wrong,mark.terminated):
              self.inPIPE.kill()
            if not self.outPIPE.state_is_(mark.wrong,mark.terminated):
              self.outPIPE.kill()
            break
          # if remote state is terminated
          elif feedback[0:1] == TerminatedMark:
            if not self.inPIPE.state_is_(mark.wrong,mark.terminated):
              self.inPIPE.stop()
            if not self.outPIPE.state_is_(mark.wrong,mark.terminated):
              self.outPIPE.stop()
            break
          # if remote state is stranded
          elif feedback[0:1] == StrandedMark:
            # stranded output PIPE
            if not self.outPIPE.state_is_( mark.stranded ):
              self.outPIPE.pause()
            time.sleep( info.TIMESCALE )
            continue
          # if remote state is active
          # we will compare the timestamp and decide the state of local
          # the same operation will be done on remote
          else:
            remoteTimeStamp = double_from_bytes(feedback[1:])
            if self.inPIPE.timestamp < remoteTimeStamp:
              self.inPIPE.activate()
              if not self.outPIPE.state_is_( mark.active ):
                self.outPIPE.activate()
            else:
              if not self.outPIPE.state_is_( mark.stranded ):
                self.outPIPE.pause()              
            continue

        elif self.inPIPE.state_is_(mark.terminated):
          if self.inPIPE.is_empty():
            # Tell the remote to stop
            _ = self.__proto.send( TerminatedMark )
            if not self.outPIPE.state_is_(mark.wrong,mark.terminated):
              self.outPIPE.stop()
            break
          else:
            # tell output PIPE
            feedback = self.__proto.send( ActiveMark + double_to_bytes(self.inPIPE.timestamp) )
            #
            if feedback[0:1] == ErrorMark:
              if not self.inPIPE.state_is_(mark.wrong,mark.terminated):
                self.inPIPE.kill()
              if not self.outPIPE.state_is_(mark.wrong,mark.terminated):
                self.outPIPE.kill()
              break
            elif feedback[0:1] == TerminatedMark:
              if not self.inPIPE.state_is_(mark.wrong,mark.terminated):
                self.inPIPE.stop()
              if not self.outPIPE.state_is_(mark.wrong,mark.terminated):
                self.outPIPE.stop()
              break
            elif feedback[0:1] == StrandedMark:
              time.sleep( info.TIMESCALE )
              continue
            else:
              packet = self.get_packet()
              self.__proto.send( PacketMark + packet.encode() ) 
        
        else:
          if self.inPIPE.is_empty():
            time.sleep( info.TIMESCALE )
            timecost += info.TIMESCALE
            if timecost > info.TIMEOUT:
              print(f"{self.name}: Timeout!")
              self.inPIPE.kill()
              self.outPIPE.kill()
              break
          else:
            # Tell remote host the state and get the feedback
            feedback = self.__proto.send( ActiveMark + double_to_bytes(self.inPIPE.timestamp) ) 
            # Check the feedback information
            # If remote state is wrong
            if feedback[0:1] == ErrorMark:
              if not self.inPIPE.state_is_(mark.wrong,mark.terminated):
                self.inPIPE.kill()
              if not self.outPIPE.state_is_(mark.wrong,mark.terminated):
                self.outPIPE.kill()
              break
            # if remote state is terminated
            elif feedback[0:1] == TerminatedMark:
              if not self.inPIPE.state_is_(mark.wrong,mark.terminated):
                self.inPIPE.stop()
              if not self.outPIPE.state_is_(mark.wrong,mark.terminated):
                self.outPIPE.stop()
              break
            # if remote state is stranded
            elif feedback[0:1] == StrandedMark:
              remoteTimeStamp = double_from_bytes( feedback[1:] )
              if remoteTimeStamp > self.inPIPE.timestamp:
                self.inPIPE.pause()
                if self.outPIPE.state_is_(mark.active):
                  self.outPIPE.pause()
                time.sleep( info.TIMESCALE )
                continue
              else:
                # stranded output PIPE
                if self.outPIPE.state_is_( mark.silent, mark.stranded ):
                  self.outPIPE.activate()
            # if remote state is decided as active
            # its ok to send packet
            else:
              packet = self.get_packet()
              self.__proto.send( PacketMark + packet.encode() ) 
          
    except Exception as e:
      try:
        self.__proto.send( ErrorMark )
      except Exception:
        pass
      raise e
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
    #
    self.link( NullPIPE() )
    
  def core_loop(self):

    self.__proto = ReceiveProtocol(bport=self.__bport)

    try:
      while True:
        
        if self.outPIPE.state_is_( mark.wrong ):
          _ = self.__proto.receive( feedback=ErrorMark )
          if not self.inPIPE.state_is_(mark.wrong,mark.terminated):
            self.inPIPE.kill()
          break

        elif self.outPIPE.state_is_( mark.terminated ):
          _ = self.__proto.receive( feedback=TerminatedMark )
          if not self.inPIPE.state_is_(mark.wrong,mark.terminated):
            self.inPIPE.stop()
          break
        
        elif self.outPIPE.state_is_( mark.stranded ):
          message = self.__proto.receive( feedback= StrandedMark + double_to_bytes(self.outPIPE.timestamp) )
          if message[0:1] == ErrorMark:
            self.outPIPE.kill()
            if self.inPIPE.state_is_(mark.wrong,mark.terminated):
              self.inPIPE.kill()
            break
          elif message[0:1] == TerminatedMark:
            self.outPIPE.stop()
            if self.inPIPE.state_is_(mark.wrong,mark.terminated):
              self.inPIPE.stop()
            break
          elif message[0:1] == StrandedMark:
            time.sleep( info.TIMESCALE )
            continue
          elif message[0:1] == ActiveMark:
            remoteTimeStamp = double_from_bytes( message[1:] )
            if self.outPIPE.timestamp < remoteTimeStamp:
              self.outPIPE.activate()
              if self.inPIPE.state_is_(mark.silent,mark.stranded):
                self.inPIPE.activate()

        else:
          message = self.__proto.receive( feedback= ActiveMark + double_to_bytes(self.inPIPE.timestamp) )
          if message[0:1] == ErrorMark:
            self.outPIPE.kill()
            if self.inPIPE.state_is_(mark.wrong,mark.terminated):
              self.inPIPE.kill()
            break
          elif message[0:1] == TerminatedMark:
            self.outPIPE.stop()
            if self.inPIPE.state_is_(mark.wrong,mark.terminated):
              self.inPIPE.stop()
            break
          elif message[0:1] == StrandedMark:
            remoteTimeStamp = double_from_bytes( message[1:] )
            if self.outPIPE.timestamp < remoteTimeStamp:
              self.outPIPE.pause()
              if self.inPIPE.state_is_(mark.active):
                self.inPIPE.pause()
            continue
          elif message[0:1] == ActiveMark:
            message = self.__proto.receive()
            assert message[0:1] == PacketMark
            packet = Packet.decode( message[1:] )
            self.put_packet( packet )

    finally:
      self.__proto.close()

  def link(self,inPIPE=None,iKey=None):
    if inPIPE is None:
      inPIPE = NullPIPE()
    super().link( inPIPE=inPIPE )

  def start(self,inPIPE=None,iKey=None):
    if inPIPE is None and self.inPIPE is None:
      inPIPE = NullPIPE()
    super().start( inPIPE=inPIPE )