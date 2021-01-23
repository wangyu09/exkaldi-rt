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
import math
import pyaudio
import wave
import time
import threading
import numpy as np
import subprocess
from collections import namedtuple

from base import StateFlag, PIPE, Packet, Element, Vector, BVector
from base import run_shell_command
from base import TIMEOUT, TIMESCALE, CMDROOT

def record(seconds=0,fileName=None):
  '''
  Record audio stream from microphone.
  The format is restricted: [ sampling rate = 16KHz, data type = int16, channels = 1 ].
  '''
  assert isinstance(seconds,(int,float)) and seconds >= 0

  rate = 16000
  width = 2
  channels = 1
  paFormat = pyaudio.paInt16
  npFormat = "int16"

  totCount = int(seconds*rate)
  perCount = 1600
  recordTimes = int(totCount/perCount)

  pa = pyaudio.PyAudio()
  stream = pa.open(format=paFormat,channels=channels,rate=rate,
              input=True,output=False)
  result = []
  try:
    if seconds == 0:
      print("Start recording!")
      while True:
        result.append(stream.read(perCount))
    else:
      print("Start recording!")
      for i in range(recordTimes):
        result.append(stream.read(perCount))
      if i * perCount < totCount:
        result.append(stream.read( (totCount-i*perCount) ) )
  except KeyboardInterrupt:
    pass
  print("Stop Recording!")

  if fileName is None:
    temp = []
    for i in result:
      i = np.frombuffer(i,dtype=npFormat)
      temp.append(i)
    return np.concatenate(temp,axis=0)
  else:
    fileName = fileName.strip()
    if not fileName.endswith(".wav"):
      fileName += ".wav"
    with wave.open(fileName, 'wb') as wf:
      wf.setnchannels(channels) 
      wf.setsampwidth(width) 
      wf.setframerate(rate) 
      wf.writeframes( b"".join(result) ) 
    return fileName

def read(fileName):
  '''
  Read audio data from wave file.
  '''
  assert os.path.isfile(fileName)

  cmd = f"./exkaldi-read-wave --file {fileName}"
  out = run_shell_command(cmd) 

  rate = int(out[0])
  points = int(out[1])
  channels = int(out[2])
  duration = float(out[3])

  data = np.array(out[4:],dtype="int16")
  if channels > 1:
    data = data.reshape([-1,channels])

  return namedtuple("WaveData",["rate","channels","points","duration","value"])(
                    rate,channels,points,duration,data
                  )

def write(waveform,fileName):
  '''
  Write waveform to file.
  The format is restricted: [ sampling rate = 16KHz, data type = int16, channels = 1 ].
  '''
  assert waveform.dtype == 'int16'
  fileName = fileName.strip()
  if not fileName.endswith(".wav"):
    fileName += ".wav"
  
  if os.path.isfile(fileName):
    os.remove(fileName)

  with wave.open(fileName, 'wb') as wf:
    wf.setnchannels(1) 
    wf.setsampwidth(2) 
    wf.setframerate(16000) 
    wf.writeframes( waveform.tobytes() ) 
  return fileName

class StreamRecorder(StateFlag):
  '''
  A class to record realtime audio stream from microphone.
  The format is restricted: [ sampling rate = 16KHz, data type = int16, channels = 1 ].
  '''
  def __init__(self,binary=True):
    super().__init__()
    assert isinstance(binary,bool), "<binary> must be bool value."

    self.__streamPIPE = PIPE()
    infos = self.__config_format(Rate=16000,Channels=1,Width=2,Points=1600)
    self.__streamPIPE.add_extra_info(info=infos)

    self.__recordThread = None
    self.__binary = binary
  
  def __config_format(self,Rate,Channels,Width,Points):
    '''
    Config wave parameters.
    '''
    assert Width in [2,4]

    self.__rate = Rate
    self.__channels = Channels
    self.__width = Width
    self.__points = Points

    if Width == 2:
      self.__paFormat = pyaudio.paInt16
      self.__format = "int16"
    else:
      self.__paFormat = pyaudio.paInt32
      self.__format = "int32"

    self.__bsize = Width * Channels * Points
    
    return namedtuple("AudioInfo",["rate","channels","width","points","bsize"])(
                                    Rate,Channels,Width,Points,self.__bsize)
  
  def get_audio_info(self):
    return self.__streamPIPE.get_extra_info()

  def get_stream_pipe(self):
    return self.__streamPIPE

  def __record_stream(self):
    '''
    The thread function to record stream from microphone.
    '''
    pa = pyaudio.PyAudio()
    stream = pa.open(format=self.__paFormat, channels=self.__channels,
                     rate=self.__rate, input=True, output=False)
    
    print("Start recording...")
    try:
      while True:
        data = stream.read(self.__points)
        if self.is_error() or self.__streamPIPE.is_error():
          self.__set_error()
          break
        elif self.is_termination() or self.__streamPIPE.is_termination():
          self.__set_termination()
          break
        else:
          if self.__binary:
            self.__streamPIPE.put( BVector(data,endpoint=False) )
          else:
            for ele in np.frombuffer(data,dtype=self.__format):
              self.__streamPIPE.put( Element(int(ele),endpoint=False) )
    except Exception as e:
      self.__set_error()
      raise e
    finally:
      stream.stop_stream()
      stream.close()
      pa.terminate()
      print("Stop recording!")
          
  def start_recording(self):
    self.__recordThread = threading.Thread(target=self.__record_stream)
    self.__recordThread.setDaemon(True)
    self.__recordThread.start()

  def __set_error(self):
    self.shift_state_to_error()
    self.__streamPIPE.set_error()

  def __set_termination(self):
    self.shift_state_to_termination()
    self.__streamPIPE.set_termination()
  
  def stop_recording(self):
    '''The main API to stop stream.'''
    self.shift_state_to_termination()

class StreamReader(StateFlag):
  '''
  A class to record realtime audio stream from wave file.
  The format is restricted: [ sampling rate = 16KHz, data type = int16, channels = 1 ].
  '''
  def __init__(self,waveFile,binary=True):
    super().__init__()
    assert os.path.isfile(waveFile), f"No such file: {waveFile}."
    assert isinstance(binary,bool)

    self.__streamPIPE = PIPE()
    with wave.open(waveFile, "rb") as wf:
      rate = wf.getframerate()
      channels = wf.getnchannels()
      width = wf.getsampwidth()
      frames = wf.getnframes()
    
    assert rate == 16000 and channels == 1 and width == 2
    self.__totalframes = frames
    points = 1600  # chunk point of each record = 1600 (0.1s)

    infos = self.__config_format(Rate=rate,Channels=channels,Width=width,Points=points)
    self.__streamPIPE.add_extra_info(info=infos)
    self.__readThread = None    
    self.__binary = binary
    self.__recource = waveFile
    self.__timeSpan = points/rate
  
  def __config_format(self,Rate,Channels,Width,Points):
    '''
    Set wave parameters.
    '''
    assert Width in [2,4]

    self.__rate = Rate
    self.__channels = Channels
    self.__width = Width
    self.__points = Points

    if Width == 2:
      self.__paFormat = pyaudio.paInt16
      self.__format = "int16"
    else:
      self.__paFormat = pyaudio.paInt32
      self.__format = "int32"

    self.__bsize = Width * Channels * Points
    
    return namedtuple("AudioInfo",["rate","channels","width","points","bsize"])(
                                    Rate,Channels,Width,Points,self.__bsize)

  def get_audio_info(self):
    return self.__streamPIPE.get_extra_info()

  def get_stream_pipe(self):
    return self.__streamPIPE

  def __read_stream(self):
    '''
    The thread function to record stream from microphone.
    '''
    readTimes = math.ceil(self.__totalframes/self.__points)
    wf = wave.open(self.__recource, "rb")

    print("Start reading...")
    try:
      for i in range(readTimes):
        st = time.time()
        if i * self.__points > self.__totalframes:
          points = self.__totalframes - i * self.__points
        else:
          points = self.__points
        data = wf.readframes(points)
        if self.is_error() or self.__streamPIPE.is_error():
          self.__set_error()
          break
        elif self.is_termination() or self.__streamPIPE.is_termination():
          self.__set_termination()
          break
        else:
          if self.__binary:
            self.__streamPIPE.put( BVector(data,endpoint=False) )
          else:
            for ele in np.frombuffer(data,dtype=self.__format):
              self.__streamPIPE.put( Element(int(ele),endpoint=False) )

        internal = self.__timeSpan - round( (time.time()-st),4)
        if internal > 0:
          time.sleep( internal )
      
    except Exception as e:
      self.__set_error()
      raise e
    else:
      if self.is_alive():
        self.__set_termination()
    finally:
      wf.close()
      print("Stop reading!")
          
  def start_reading(self):
    self.__readThread = threading.Thread(target=self.__read_stream)
    self.__readThread.setDaemon(True)
    self.__readThread.start()

  def __set_error(self):
    self.shift_state_to_error()
    self.__streamPIPE.set_error()

  def __set_termination(self):
    self.shift_state_to_termination()
    self.__streamPIPE.set_termination()
  
  def stop_reading(self):
    '''The main API to stop stream.'''
    self.shift_state_to_termination()

class FrameCutter(StateFlag):
  '''
  A class to cut frame.
  '''
  def __init__(self,width=400,shift=160):
    super().__init__()
    assert isinstance(width,int) and isinstance(shift,int)
    assert 0 < shift < width
    self.__width = width
    self.__shift = shift
    self.__cover = width - shift
    self.__streamBuffer = np.zeros([width,],dtype="int16")
    self.__framePIPE = PIPE()

    self.__reset_position_flag()
  
  def __reset_position_flag(self):
    self.__zerothStep = True
    self.__terminationStep = False

  def get_window_info(self):
    return namedtuple("WindowInfo",["width","shift"])(
                            self.__width,self.__shift)

  def __set_error(self):
    self.shift_state_to_error()
    self.__framePIPE.set_error()
  
  def __set_termination(self):
    self.shift_state_to_termination()
    self.__framePIPE.set_termination()

  def __prepare_frame_stream(self,streamPIPE):
    '''
    Prepare chunk stream to compute feature.
    '''
    timeCost = 0
    addedNewData = False 

    # retain old
    if self.__zerothStep:
      pos = 0
      self.__zerothStep = False
    else:
      self.__streamBuffer[0:self.__cover] = self.__streamBuffer[ -self.__cover: ]
      pos = self.__cover

    while pos < self.__width:
      if streamPIPE.is_error():
        self.__set_error()
        return False
      elif streamPIPE.is_exhaustion():
        self.__terminationStep = True
        break
      elif streamPIPE.is_empty():
        time.sleep(TIMESCALE)
        timeCost += TIMESCALE
        if timeCost > TIMEOUT:
          streamPIPE.set_error()
          self.__set_error()
          return False
      else:
        ele = streamPIPE.get()
        assert isinstance(ele,Element)
        if addedNewData:
          self.__streamBuffer[pos] = ele.item
          pos += 1
          if ele.is_endpoint():
            self.__terminationStep = True
            break
        else:
          if ele.is_endpoint():
            continue # discard this element
          self.__streamBuffer[pos] = ele.item
          pos += 1
          addedNewData = True
    
    if pos == 0:
      self.__set_termination()
      return False
    else:
      self.__streamBuffer[pos:] = 0
    
    return True   

  def __cut_frame(self,streamPIPE):
    print("Start cutting frames...")
    try:
      while True:
        # preprare chunk stream
        if not self.__prepare_frame_stream(streamPIPE):
          break
        # add to PIPE
        if not self.__framePIPE.is_alive():
          streamPIPE.set_error()
          self.__set_error()
          break
        else:
          if self.__terminationStep:
            self.__framePIPE.put( Vector(self.__streamBuffer.copy(), endpoint=True) )
            self.__reset_position_flag()
          else:
            self.__framePIPE.put( Vector(self.__streamBuffer.copy(), endpoint=False) )
        # if no more data
        if streamPIPE.is_exhaustion():
          self.__set_termination()
          break
    except Exception as e:
      streamPIPE.set_error()
      self.__set_error()
      raise e
    finally:
      print("Stop cutting frames!")
  
  def start_cutting(self,streamPIPE):
    self.__cutThread = threading.Thread(target=self.__cut_frame,args=(streamPIPE,))
    self.__cutThread.setDaemon(True)
    self.__cutThread.start()

  def get_frame_pipe(self):
    return self.__framePIPE

class ActivityDetector(StateFlag):

  def __init__(self,frameDim=400,chunkFrames=10):
    super().__init__()
    assert isinstance(frameDim,int) and isinstance(chunkFrames,int)
    assert frameDim > 0 and chunkFrames > 0

    self.__chunkFrames = chunkFrames
    self.__frameBuffer = np.zeros([chunkFrames,frameDim],dtype="int16")
    self.__newFramePIPE = PIPE()

    self.vad_function = None
    self.__detectThread = None
    self.__reset_position_flag()
  
  def __reset_position_flag(self):
    self.__terminationStep = False
    self.__avaliableFrames = self.__chunkFrames

  def __set_error(self):
    self.shift_state_to_error()
    self.__newFramePIPE.set_error()
  
  def __set_termination(self):
    self.shift_state_to_termination()
    self.__newFramePIPE.set_termination()
  
  def __prepare_chunk_frame(self,framePIPE):

    timeCost = 0
    pos = 0

    while pos < self.__chunkFrames:
      if framePIPE.is_error():
        self.__set_error()
        return False
      elif framePIPE.is_exhaustion():
        self.__avaliableFrames = pos
        self.__terminationStep = True
        break
      elif framePIPE.is_empty():
        time.sleep(TIMESCALE)
        timeCost += TIMESCALE
        if timeCost > TIMEOUT:
          framePIPE.set_error()
          self.__set_error()
          return False
      else:
        vec = framePIPE.get()
        assert isinstance(vec,Vector)
        if pos != 0:
          self.__frameBuffer[pos,:] = vec.item
          pos += 1
          if vec.is_endpoint():
            self.__terminationStep = True
            self.__avaliableFrames = pos
            break
        else:
          if vec.is_endpoint():
            continue # discard this element
          self.__frameBuffer[pos,:] = vec.item
          pos += 1
    
    if pos == 0:
      self.__set_termination()
      return False
    else:
      self.__frameBuffer[pos:,:] = 0
    
    return True

  def __detect(self,framePIPE):
    print("Start voice activity detecting...")
    try:
      while True:
        # try to prepare chunk frames
        self.__frameBuffer.flags.writeable = True
        if not self.__prepare_chunk_frame(framePIPE):
          break
        self.__frameBuffer.flags.writeable = False
        # detect
        if not self.__terminationStep:
          result = self.vad_function(self.__frameBuffer[0:self.__avaliableFrames,:])
        # add to new PIPE
        if not self.__newFramePIPE.is_alive():
          framePIPE.set_error()
          self.__set_error()
          break
        else:
          if self.__terminationStep:
            for fid in range(self.__avaliableFrames-1):
              self.__newFramePIPE.put( Vector(self.__frameBuffer[fid],endpoint=False) )
            self.__newFramePIPE.put( Vector(self.__frameBuffer[self.__avaliableFrames-1],endpoint=True) )
            self.__reset_position_flag()
          elif result is True:
            for fid in range(self.__avaliableFrames):
              self.__newFramePIPE.put( Vector(self.__frameBuffer[fid],endpoint=False) )
          else:
            self.__newFramePIPE.put( Vector(self.__frameBuffer[0],endpoint=True) )
        # if no more data
        if framePIPE.is_exhaustion():
          self.__set_termination()
          break
    except Exception as e:
      framePIPE.set_error()
      self.__set_error()
      raise e
    finally:
      print("Stop voice activity detecting!")

  def start_detecting(self,framePIPE):

    assert self.vad_function is not None, "Please implement vad function."
    result = self.vad_function(self.__frameBuffer)
    assert isinstance(result,bool)

    self.__detectThread = threading.Thread(target=self.__detect,args=(framePIPE,))
    self.__detectThread.setDaemon(True)
    self.__detectThread.start()

  def get_frame_pipe(self):
    return self.__newFramePIPE
