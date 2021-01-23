# coding=utf-8
#
# Yu Wang (University of Yamanashi)
# Dec, 2020
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

import socket
from base import TIMEOUT, StateDescriptor, PIPE

socket.setdefaulttimeout(TIMEOUT)

class TCPClient(StateDescriptor):
  '''
  An object running on local client.
  '''
  def __init__(self,thost,tport):
    super().__init__()
    self.__client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    print("Try to connect to server...")
    self.__client.connect((thost,tport),)
    print("Connect Done!")

    self.__resultPIPE = PIPE()
    self.__chunkSize = None

    self.__sendThread = None
    self.__receiveThread = None

  def get_result_pipe(self):
    return self.__resultPIPE

  def __send(self,streamPIPE,timeout):
    '''The thread function to send stream data to server'''
    timecost = 0
    while True:
      if self.is_error() or streamPIPE.is_error() or (not self.__resultPIPE.is_alive()):
        streamPIPE.kill()
        self.__sending_kill()
        break
      elif self.is_termination() or streamPIPE.is_exhaustion():
        streamPIPE.terminate()
        self.__sending_terminate()
        break
      elif streamPIPE.is_empty():
        time.sleep(0.1)
        timecost += 0.1
        if timecost > timeout:
          streamPIPE.kill()
          self.__sending_kill()
          break
      else:
        data = streamPIPE.get()
        self.__client.send(data)

  def __sending_kill(self):
    self.shift_state_to_error()
    self.__client.send(b"errFlag")

  def __sending_terminate(self):
    self.__client.send(b"endFlag")

  def start_sending(self,streamPIPE):

    # send the wave information
    extraInfo = streamPIPE.get_extra_info()
    self.__chunkSize = extraInfo[-1]

    firstMess = str(extraInfo)
    firstMess = (firstMess + (32-len(firstMess))*" ").encode()
    self.__client.send(firstMess)

    # Start sending thread
    self.__sendThread = threading.Thread(target=self.__send,args=(streamPIPE,TIMEOUT,))
    self.__sendThread.setDaemon(True)
    self.__sendThread.start()

  def __receive(self):
    while True:
      if self.is_error() or self.__resultPIPE.is_error():
        self.__receiving_kill()
        break
      elif self.is_termination() or self.__resultPIPE.is_termination():
        self.__receiving_terminate()
        break 
      else:
        message = self.__client.recv(self.__chunkSize)
        if message == b"errFlag":
          self.__receiving_kill()
          break
        elif message == b"endFlag":
          self.__receiving_terminate()
          break
        else:
          self.__resultPIPE.put(message)
  
  def __receiving_kill(self):
    self.shift_state_to_error()
    self.__resultPIPE.kill()
    self.__client.close()

  def __receiving_terminate(self):
    self.shift_state_to_termination()
    self.__resultPIPE.terminate()
    self.__client.close()

  def start_receiving(self):
    self.__receiveThread = threading.Thread(target=self.__receive)
    self.__receiveThread.setDaemon(True)
    self.__receiveThread.start()

  def debug(self):
    return self.__sendThread.is_alive()

class TCPServer(StateDescriptor):

  def __init__(self,bindHost,bindPort):
    super().__init__()
    # connect the client
    self.__server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    self.__server.bind((bindHost,bindPort),)
    print("Listening to connect...")
    self.__server.listen(1)
    self.__client, _ = self.__server.accept()
    print("Connected!")
    #
    self.__streamPIPE = PIPE()
    self.__receiveThread = None
    self.__sendThread = None

    self.__chunkSize = None

  def get_stream_pipe(self):
    return self.__streamPIPE

  def __receive(self):
    '''
    The thread function to receive stream from client.
    '''
    while True:
      if self.is_error() or self.__streamPIPE.is_error():
        self.__receiving_kill()
        break
      elif self.is_termination() or self.__streamPIPE.is_termination():
        self.__receiving_terminate()
        break
      else:
        message = self.__client.recv(self.__chunkSize)
        if message == b"errFlag":
          self.__receiving_kill()
          break
        elif message == b"endFlag":
          self.__receiving_terminate()
          break
        else:
          self.__streamPIPE.puts( np.frombuffer(message,dtype=self.__format) )

  def __receiving_kill(self):
    '''
    When received error flag.
    '''
    self.shift_state_to_error()
    self.__streamPIPE.kill()
    self.__client.close()
  
  def __sending_kill(self):
    '''
    When error occured on server
    '''
    self.shift_state_to_error()
    self.__client.send(b"errFlag")
    self.__client.close()

  def __receiving_terminate(self):
    '''
    When received end flag
    '''
    self.__streamPIPE.terminate()
  
  def __sending_terminate(self):
    '''
    When recognition is over
    '''
    self.shift_state_to_termination()
    self.__client.send(b"endFlag")
    self.__client.close()

  def start_receiving(self):
    '''
    Receive stream from client.
    '''
    vert = self.__client.recv(32)
    vert = vert.decode().strip().strip("()").split(',')
    Rate = int(vert[0])
    Channels = int(vert[1])
    Width = int(vert[2])
    Points = int(vert[3])
    Size = int(vert[4])
    self.__format = "int"+ str(Width*8)
    self.__streamPIPE.add_extra_info(items=(Rate,Channels,Width,Points,Size))
    self.__chunkSize = Size

    self.__receiveThread = threading.Thread(target=self.__receive)
    self.__receiveThread.setDaemon(True)
    self.__receiveThread.start()

  def __send(self,resultPIPE,timeout):
    '''
    The thread function to send result to client.
    '''
    timecost = 0
    while True:
      if self.is_error() or resultPIPE.is_error():
        resultPIPE.kill()
        self.__sending_kill()
        break
      elif self.is_termination() or resultPIPE.is_exhaustion():
        resultPIPE.terminate()
        self.__sending_terminate()
        break
      elif resultPIPE.is_empty():
        time.sleep(0.1)
        timecost += 0.1
        if timecost > timeout:
          resultPIPE.kill()
          self.__sending_kill()
          break
      else:
        message = resultPIPE.get()
        self.__client.send(message)
        timecost = 0
  
  def start_sending(self,resultPIPE):
    '''
    Send the result.
    '''
    self.__sendThread = threading.Thread(target=self.__send,args=(resultPIPE,TIMEOUT,))
    self.__sendThread.setDaemon(True)
    self.__sendThread.start()

  def debug(self):
    return self.__receiveThread.is_alive()
