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
import math
import pyaudio
import wave
import time
import threading
import webrtcvad
import numpy as np
import subprocess
from collections import namedtuple

from exkaldi2.base import Component, PIPE, Packet, Element, Vector, BVector
from exkaldi2.base import run_exkaldi_shell_command
from exkaldi2.base import info

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
      if (i + 1) * perCount < totCount:
        result.append(stream.read( (totCount-(i+1)*perCount) ) )
  except KeyboardInterrupt:
    pass
  print("Stop Recording!")

  if fileName is None:
    content = [ np.frombuffer(i,dtype=npFormat) for i in result ]
    content = np.concatenate(content,axis=0)
    points = len(content)
    duration = round(points/rate,2)
    return namedtuple("Wave",["rate","channels","points","duration","value"])(
                    rate,channels,points,duration,content
                  )
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
  assert os.path.isfile(fileName), f"No such file: {fileName}"

  cmd = os.path.join(info.CMDROOT,f"exkaldi-read-wave --file {fileName}")
  out = run_exkaldi_shell_command(cmd) 

  rate = int(out[0])
  points = int(out[1])
  channels = int(out[2])
  duration = float(out[3])

  data = np.array(out[4:],dtype="int16")
  if channels > 1:
    data = data.reshape([-1,channels])

  return namedtuple("Wave",["rate","channels","points","duration","value"])(
                    rate,channels,points,duration,data
                  )

def write(waveform,fileName):
  '''
  Write waveform to file.
  The format is restricted: [ sampling rate = 16KHz, data type = int16, channels = 1 ].
  '''
  assert waveform.dtype == 'int16', "We only support writing int16 wave data."
  fileName = fileName.strip()
  if not fileName.endswith(".wav"):
    fileName += ".wav"
  
  if os.path.isfile(fileName):
    os.remove(fileName)

  with wave.open(fileName,'wb') as wf:
    wf.setnchannels(1) 
    wf.setsampwidth(2) 
    wf.setframerate(16000) 
    wf.writeframes( waveform.tobytes() ) 
  return fileName

class StreamRecorder(Component):
  '''
  A class to record realtime audio stream from microphone.
  The format has been restricted: [ sampling rate = 16KHz, data type = int16, channels = 1 ].
  '''
  def __init__(self,binary=True,name="recorder"):
    super().__init__(name=name)
    assert isinstance(binary,bool), "<binary> must be bool value."

    self.__streamPIPE = PIPE()
    infos = self.__config_format(Rate=16000,Channels=1,Width=2,Points=1600)
    self.__streamPIPE.add_extra_info(info=infos)

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

    return namedtuple("AudioInfo",["rate","channels","width"])(
                                    Rate,Channels,Width)
  
  def get_audio_info(self):
    return self.__streamPIPE.get_extra_info()

  @property
  def outPIPE(self):
    '''Return the stream PIPE'''
    return self.__streamPIPE

  def __record_stream(self):
    '''
    The thread function to record stream from microphone.
    '''
    pa = pyaudio.PyAudio()
    stream = pa.open(format=self.__paFormat,channels=self.__channels,
                     rate=self.__rate,input=True,output=False)
    
    print("Start recording...")
    try:
      while True:
        data = stream.read(self.__points)
        if self.is_error() or self.__streamPIPE.is_error():
          self.kill()
          break
        elif self.is_termination() or self.__streamPIPE.is_termination():
          self.stop()
          break
        else:
          if self.__binary:
            self.__streamPIPE.put( BVector(data,dtype=self.__format,endpoint=False) )
          else:
            for ele in np.frombuffer(data,dtype=self.__format):
              self.__streamPIPE.put( Element(ele,endpoint=False) )
    except Exception as e:
      self.kill()
      raise e
    finally:
      stream.stop_stream()
      stream.close()
      pa.terminate()
      print("Stop recording!")
          
  def _start(self,inPIPE=None):
    '''<inPIPE> is just a place holder'''
    recordThread = threading.Thread(target=self.__record_stream)
    recordThread.setDaemon(True)
    recordThread.start()
    return recordThread

class StreamReader(Component):
  '''
  A class to record realtime audio stream from wave file.
  The format is restricted: [ sampling rate = 16KHz, data type = int16, channels = 1 ].
  '''
  def __init__(self,waveFile,binary=True,name="reader"):
    super().__init__(name=name)
    assert os.path.isfile(waveFile), f"No such file: {waveFile}."
    assert isinstance(binary,bool)

    self.__streamPIPE = PIPE()
    with wave.open(waveFile, "rb") as wf:
      rate = wf.getframerate()
      channels = wf.getnchannels()
      width = wf.getsampwidth()
      frames = wf.getnframes()
    
    assert rate == 16000 and channels == 1 and width == 2, \
          "We only support such wave file with format of audio sampling rate = 16KHz, channels = 1 and int16."
    self.__totalframes = frames
    points = 1600  # chunk point of each record = 1600 (0.1s)

    infos = self.__config_format(Rate=rate,Channels=channels,Width=width,Points=points)
    self.__streamPIPE.add_extra_info(info=infos)  
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
    
    return namedtuple("AudioInfo",["rate","channels","width"])(
                                    Rate,Channels,Width)

  def get_audio_info(self):
    return self.__streamPIPE.get_extra_info()

  @property
  def outPIPE(self):
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
          self.kill()
          break
        elif self.is_termination() or self.__streamPIPE.is_termination():
          self.stop()
          break
        else:
          if self.__binary:
            self.__streamPIPE.put( BVector(data,dtype=self.__format,endpoint=False) )
          else:
            for ele in np.frombuffer(data,dtype=self.__format):
              self.__streamPIPE.put( Element(ele,endpoint=False) )

        internal = self.__timeSpan - round( (time.time()-st),4)
        if internal > 0:
          time.sleep( internal )
      
    except Exception as e:
      self.kill()
      raise e
    else:
      if self.is_alive():
        self.stop()
    finally:
      wf.close()
      print("Stop reading!")
          
  def _start(self,inPIPE=None):
    '''__inPIPE__ is just a placeholder.'''
    readThread = threading.Thread(target=self.__read_stream)
    readThread.setDaemon(True)
    readThread.start()
    return readThread

class FrameCutter(Component):
  '''
  A class to cut frame.
  '''
  def __init__(self,width=400,shift=160,name="cutter"):
    super().__init__(name=name)
    assert isinstance(width,int) and isinstance(shift,int)
    assert 0 < shift <= width
    self.__width = width
    self.__shift = shift
    self.__cover = width - shift
    self.__streamBuffer = np.zeros([width,],dtype="int16")
    self.__framePIPE = PIPE()
    self.__elementCache = PIPE()

    self.__cutThread = None
    self.__reset_position_flag()
  
  def __reset_position_flag(self):
    self.__zerothStep = True
    self.__terminationStep = False

  def get_window_info(self):
    return namedtuple("WindowInfo",["width","shift"])(
                            self.__width,self.__shift)

  @property
  def outPIPE(self):
    return self.__framePIPE

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
      self.__streamBuffer[0:self.__cover] = self.__streamBuffer[ self.__shift: ]
      pos = self.__cover

    while pos < self.__width:
      if not self.__elementCache.is_empty():
        ele = self.__elementCache.get()
        if addedNewData:
          self.__streamBuffer[pos] = ele.data
          pos += 1
          if ele.is_endpoint():
            self.__terminationStep = True
            break
        else:
          if ele.is_endpoint():
            continue # discard this element
          self.__streamBuffer[pos] = ele.data
          pos += 1
          addedNewData = True
      else:
        if streamPIPE.is_error():
          self.kill()
          return False
        elif streamPIPE.is_exhaustion():
          self.__terminationStep = True
          break
        elif streamPIPE.is_empty():
          time.sleep(info.TIMESCALE)
          timeCost += info.TIMESCALE
          if timeCost > info.TIMEOUT:
            streamPIPE.kill()
            self.kill()
            return False
        else:
          pac = streamPIPE.get()
          if isinstance(pac,Element):
            self.__elementCache.put(pac)
            continue
          elif isinstance(pac,(Vector,BVector)):
            if isinstance(pac,BVector):
              pac = pac.decode()
            data = pac.data
            if pac.is_endpoint():
              self.__elementCache.put( Element(data[0],endpoint=True) )
            else:
              for ele in data:
                self.__elementCache.put( Element(ele,endpoint=False) )
          else:
            raise Exception(f'Unknown packet: {type(pac).__name__}')
          
    if pos == 0:
      self.stop()
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
          streamPIPE.kill()
          self.kill()
          break
        else:
          if self.__terminationStep:
            self.__framePIPE.put( Vector(self.__streamBuffer.copy(), endpoint=True) )
            self.__reset_position_flag()
          else:
            self.__framePIPE.put( Vector(self.__streamBuffer.copy(), endpoint=False) )
        # if no more data
        if streamPIPE.is_exhaustion():
          self.stop()
          break
    except Exception as e:
      streamPIPE.kill()
      self.kill()
      raise e
    finally:
      print("Stop cutting frames!")
  
  def _start(self,inPIPE):
    '''<inPIPE> should be a stream PIPE'''
    cutThread = threading.Thread(target=self.__cut_frame,args=(inPIPE,))
    cutThread.setDaemon(True)
    cutThread.start()
    return cutThread

class ActivityDetector(Component):

  def __init__(self,inDim=0,batchSize=480,vadMode=3,name="detector"):
    super().__init__(name=name)
    assert isinstance(inDim,int) and isinstance(batchSize,int)
    assert inDim >= 0 and batchSize > 0
    if inDim == 0:
      self.__outClass = Element
      self.__frameBuffer = np.zeros([batchSize,],dtype="int16")
    else:
      self.__outClass = Vector
      self.__frameBuffer = np.zeros([batchSize,inDim],dtype="int16")

    self.__batchSize = batchSize
    self.__outPIPE = PIPE()

    self.vad_function = None
    self.__detectThread = None
    self.__reset_position_flag()

    self.__detectThread = None

    assert vadMode in [1,2,3], f"<vadMode> must be 1, 2 or 3 but got: {vadMode}."
    self.__webrtcvadobj = webrtcvad.Vad(vadMode)
  
  def __default_vad_function(self,chunk):
    assert len(chunk.shape) == 1
    assert len(chunk) in [160,320,480]
    return self.__webrtcvadobj.is_speech(chunk.tobytes(),16000)

  def __reset_position_flag(self):
    self.__terminationStep = False
    self.__avaliableFrames = self.__batchSize

  def __prepare_chunk_frame(self,inPIPE):

    timeCost = 0
    pos = 0

    while pos < self.__batchSize:
      if inPIPE.is_error():
        self.kill()
        return False
      elif inPIPE.is_exhaustion():
        self.__avaliableFrames = pos
        self.__terminationStep = True
        break
      elif inPIPE.is_empty():
        time.sleep(info.TIMESCALE)
        timeCost += info.TIMESCALE
        if timeCost > info.TIMEOUT:
          inPIPE.kill()
          self.kill()
          return False
      else:
        vec = inPIPE.get()
        assert isinstance(vec,(Element,Vector)), f"Activity Detector needs Element or Vector packet but got: {type(vec).__name__}."
        if pos != 0:
          self.__frameBuffer[pos] = vec.data
          pos += 1
          if vec.is_endpoint():
            self.__terminationStep = True
            self.__avaliableFrames = pos
            break
        else:
          if vec.is_endpoint():
            continue # discard this element
          self.__frameBuffer[pos] = vec.data
          pos += 1
    
    if pos == 0:
      self.stop()
      return False
    else:
      self.__frameBuffer[pos:] = 0
    
    return True

  def __detect(self,inPIPE):
    print("Start voice activity detecting...")
    try:
      while True:
        # try to prepare chunk frames
        self.__frameBuffer.flags.writeable = True
        if not self.__prepare_chunk_frame(inPIPE):
          break
        self.__frameBuffer.flags.writeable = False
        # detect
        if not self.__terminationStep:
          result = self.vad_function(self.__frameBuffer[0:self.__avaliableFrames])
        # add to new PIPE
        if not self.__outPIPE.is_alive():
          inPIPE.kill()
          self.kill()
          break
        else:
          if self.__terminationStep:
            for fid in range(self.__avaliableFrames-1):
              self.__outPIPE.put( self.__outClass(self.__frameBuffer[fid],endpoint=False) )
            self.__outPIPE.put( self.__outClass(self.__frameBuffer[self.__avaliableFrames-1],endpoint=True) )
            self.__reset_position_flag()
          elif result is True:
            for fid in range(self.__avaliableFrames):
              self.__outPIPE.put( self.__outClass(self.__frameBuffer[fid],endpoint=False) )
          else:
            self.__outPIPE.put( self.__outClass(self.__frameBuffer[0],endpoint=True) )
            #for fid in range(self.__avaliableFrames):
            #  self.__outPIPE.put( self.__outClass(self.__frameBuffer[fid],endpoint=True) )
        # if no more data
        if inPIPE.is_exhaustion():
          self.stop()
          break
    except Exception as e:
      inPIPE.kill()
      self.kill()
      raise e
    finally:
      print("Stop voice activity detecting!")

  def _start(self,inPIPE):

    if self.vad_function is None:
      self.vad_function = self.__default_vad_function
    else:
      result = self.vad_function(self.__frameBuffer)
      assert isinstance(result,bool)

    detectThread = threading.Thread(target=self.__detect,args=(inPIPE,))
    detectThread.setDaemon(True)
    detectThread.start()

    return detectThread

  @property
  def outPIPE(self):
    return self.__outPIPE


