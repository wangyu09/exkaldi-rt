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

from exkaldi_rt.base import ExKaldi2Base, Component, PIPE, Packet, Element, Vector
from exkaldi_rt.base import run_exkaldi_shell_command
from exkaldi_rt.base import info, KillableThread
from exkaldi_rt.base import ENDPOINT, is_endpoint

def record(seconds=0,fileName=None):
  '''
  Record audio stream from microphone.
  The format is restricted: sampling rate -> 16KHz, data type -> int16, channels -> 1.

  Args:
    _seconds_: an int value. If == 0, do not limit time.
    _fileName_: a string. If None, return wave data.
  
  Return:
    a string: If _fileName_ is a path.
    A Wave object: hold wave format and data. 
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

def cut_frames(waveform,width,shift):
  points = len(waveform)
  assert points >= width
  
  N = int((points-width)/shift) + 1
  result = np.zeros([N,width],dtype="int16")
  for i in range(N):
    offset = i * shift
    result[i] = waveform[ offset:offset+width ]
  
  return result

class VADetector(ExKaldi2Base):
  '''
  Voice Activity Detector used to be embeded in StreamReader or StreamRecorder.
  Note that this is not Component.
  '''
  def __init__(self,patience=7,truncate=False,name=None):
    # Initialize state and name
    super().__init__(name=name)
    # The maximum continuous ENDPOINT length.
    self.__patience = patience
    #
    assert isinstance(truncate,bool)
    self.__truncate = truncate
    # Config others
    self.reset()
    # core function
    self.is_speech = None

  def reset(self):
    self.__silenceCounter = 0

  def detect(self,chunk):
    '''
    Return:
      True: This chunk data should be retained.
      False: This chunk data should be discarded.
      None: This chunk data should be replaced with ENDPOINT.
    '''
    assert self.is_speech is not None, f"{self.name}: Please implement .is_speech function."
    assert isinstance(chunk,bytes)
    activity = self.is_speech(chunk)
    # If this is not silence
    if activity:
      self.__silenceCounter = 0
      return True
    # If this is silence
    else:
      self.__silenceCounter += 1
      if (self.__silenceCounter == self.__patience):
        return None if self.__truncate else False
      elif self.__silenceCounter > self.__patience:
        return False
      else:
        return True

class WebrtcVADetector(VADetector):

  def __init__(self,patience=7,mode=3,truncate=False,name=None):
    # Initialize
    super().__init__(patience=patience,truncate=truncate,name=name)
    # Define a webrtcvad object.
    assert mode in [1,2,3], f"{self.name}: <mode> must be 1, 2 or 3, but got: {mode}."
    self.__webrtcvadobj = webrtcvad.Vad(mode)
    self.is_speech = self.__is_speech

  def __is_speech(self,chunk)->bool:
    return self.__webrtcvadobj.is_speech(chunk,16000)

"""
class StreamRecorder(Component):
  '''
  A class to record realtime audio stream from microphone.
  The format has been restricted: [ sampling rate = 16KHz, data type = int16, channels = 1 ].
  '''
  def __init__(self,name="recorder"):
    super().__init__(name=name)

    self.outPIPE = PIPE()
    infos = self.__config_format(Rate=16000,Channels=1,Width=2,Points=1600)
    self.outPIPE.add_extra_info(info=infos)
  
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
    return self.outPIPE.get_extra_info()

  @property
  def outPIPE(self):
    '''Return the stream PIPE'''
    return self.outPIPE

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
        if self.is_wrong() or self.outPIPE.is_wrong():
          self.kill()
          break
        elif self.outPIPE.is_terminated():
          self.stop()
          break
        else:
          for ele in np.frombuffer(data,dtype=self.__format):
            self.outPIPE.put( Element(ele) )
          if self.is_terminated():
            self.outPIPE.put( ENDPOINT )
            self.stop()
            break
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
"""

class StreamReader(Component):
  '''
  A class to record realtime audio stream from wave file.
  The format is restricted: [ sampling rate = 16KHz, data type = int16, channels = 1 ].
  '''
  def __init__(self,waveFile,chunkSize=480,simulate=True,vaDetector=None,name=None):
    super().__init__(name=name)

    # Config audio infomation
    infos = self.__direct_source(waveFile)
    self.outPIPE.add_extra_info(info=infos)  

    # If simulate is True, simulate real-time reading.
    assert isinstance(simulate,bool), f"{self.name}: <simulate> should be a bool value."
    self.__simulate = simulate
    
    # Config chunk size
    assert isinstance(chunkSize,int) and chunkSize > 0
    self.__points = chunkSize

    # Config VAD
    self.__vad = None
    if vaDetector is not None:
      assert isinstance(vaDetector,VADetector), f"{self.name}: <vaDetector> should be a VADetector object."
      if isinstance(vaDetector,WebrtcVADetector):
        assert self.__points in [160,320,480], f"{self.name}: If use webrtcvad, please set the chunk size: 160, 320,or 480."
      self.__vad = vaDetector
    
    # A flag for time sleep
    self.__timeSpan = self.__points/self.__rate
  
  def __direct_source(self,waveFile):
    assert os.path.isfile(waveFile), f"{self.name}: No such file: {waveFile}."
    # Get audio info
    with wave.open(waveFile, "rb") as wf:
      rate = wf.getframerate()
      channels = wf.getnchannels()
      width = wf.getsampwidth()
      frames = wf.getnframes()
    assert rate == 16000 and channels == 1 and width == 2, \
          "We only support the audio file with format of sampling rate = 16KHz, channels = 1 and width = 2."
    self.__totalframes = frames
    self.__recource = waveFile
    return self.__config_format(rate,channels,width,frames,frames/rate)

  def redirect(self,waveFile):
    if self.is_alive():
      raise Exception(f"{self.name}: Can not redirect a wave file during reader is running!")
    # Reset this component firstly
    self.reset()
    infos = self.__direct_source(waveFile)
    self.outPIPE.add_extra_info(info=infos) 

  def __config_format(self,Rate,Channels,Width,Frames,Duration):
    '''
    Set wave parameters.
    '''
    assert Width in [2,4]

    self.__rate = Rate
    self.__channels = Channels
    self.__width = Width

    if Width == 2:
      self.__paFormat = pyaudio.paInt16
      self.__format = "int16"
    else:
      self.__paFormat = pyaudio.paInt32
      self.__format = "int32"
    
    return namedtuple("AudioInfo",["rate","channels","width","frames","duration"])(
                                    Rate,Channels,Width,Frames,Duration)

  def get_audio_info(self):
    '''
    Return the audio information.
    '''
    return self.outPIPE.get_extra_info()

  def __read_stream(self):
    '''
    The thread function to record stream from microphone.
    '''
    readTimes = math.ceil(self.__totalframes/self.__points)
    wf = wave.open(self.__recource, "rb")

    print(f"{self.name}: Start...")
    try:
      for i in range(readTimes):
        st = time.time()
        # read a chunk of stream
        data = wf.readframes(self.__points)
        # detcet if necessary
        if self.__vad is not None:
          if len(data) != self.__width*self.__points:
            data += np.zeros( (self.__width*self.__points-len(data))//2, dtype="int16" ).tobytes() 
          valid = self.__vad.detect(data)
        else:
          valid = True

        # add data
        if valid is True:
          # check pipe state and append data
          if self.is_wrong() or \
            self.outPIPE.is_wrong() or \
            self.outPIPE.is_terminated():
            self.kill()
            break
          else:
            ## append data
            for ele in np.frombuffer(data,dtype=self.__format):
              self.outPIPE.put( Element(ele) )
        elif valid is None:
          self.outPIPE.put( ENDPOINT )

        ## if reader has been stopped by force
        if self.is_terminated():
          self.stop()
          break

        # wait if necessary
        if self.__simulate:
          internal = self.__timeSpan - round( (time.time()-st),4)
          if internal > 0:
            time.sleep( internal )

    except Exception as e:
      self.kill()
      raise e

    else:
      # If reader stopped normally
      if self.is_alive():
        self.stop()
    finally:
      wf.close()
      print(f"{self.name}: Stop!")
          
  def _start(self,inPIPE=None):
    '''__inPIPE__ is just a placeholder.'''
    readThread = KillableThread(target=self.__read_stream)
    readThread.setDaemon(True)
    readThread.start()
    return readThread

  def start(self,inPIPE=None):
    '''_inPIPE_ is just a place holder'''
    super().start(None)

class ElementFrameCutter(Component):
  '''
  A class to cut frame from Element PIPE of Vector PIPE.
  '''
  def __init__(self,width=400,shift=160,name=None):
    super().__init__(name=name)
    # Config some size parameters
    assert isinstance(width,int) and isinstance(shift,int)
    assert 0 < shift <= width
    self.__width = width
    self.__shift = shift
    self.__cover = width - shift
    # Prepare a work buffer
    self.__streamBuffer = np.zeros([self.__width,],dtype="int16")
    # Config some position flags
    self.__reset_position_flag()

  def reset(self):
    super().reset()
    # Clear the stream buffer
    self.__streamBuffer *= 0
    # Reset position flags
    self.__reset_position_flag()

  def __reset_position_flag(self):
    '''
    Some flags to mark index.
    '''
    self.__zerothStep = True
    self.__firstStep = False
    self.__endpointStep = False
    self.__finalStep = False
    self.__tailIndex = self.__width

  def get_window_info(self):
    '''
    Get the window information.
    '''
    return namedtuple("WindowInfo",["width","shift"])(
                            self.__width,self.__shift)

  def __prepare_frame_stream(self,streamPIPE):
    '''
    Prepare chunk stream to compute feature.
    '''
    timecost = 0
    # copy old data if necessary
    if self.__zerothStep:
      pos = 0
      self.__zerothStep = False
      self.__firstStep = True
    else:
      self.__streamBuffer[0:self.__cover] = self.__streamBuffer[self.__shift:]
      pos = self.__cover
      self.__firstStep = False
    # get new data
    while pos < self.__width:
      ## If error occurred in stream PIPE
      if streamPIPE.is_wrong():
        self.kill()
        return False
      ## If no more data
      elif streamPIPE.is_exhausted():
        self.__finalStep = True
        self.__tailIndex = pos
        break
      ## If need wait because of receiving no data
      elif streamPIPE.is_empty():
        time.sleep(info.TIMESCALE)
        timecost += info.TIMESCALE
        if timecost > info.TIMEOUT:
          print(f"{self.name}: Timeout! Did not receive any data for a long time！")
          # Try to kill stream PIPE
          streamPIPE.kill()
          # Kill self 
          self.kill()
          return False
      ## If need wait because of blocked
      elif streamPIPE.is_blocked():
        time.sleep(info.TIMESCALE)
      ## If had data
      else:
        ele = streamPIPE.get()
        if is_endpoint(ele):
          self.__endpointStep = True
          self.__tailIndex = pos
          break
        else:
          assert isinstance(ele,Element), f"{self.name}: Need Element packet but got: {type(ele).__name__}"
          self.__streamBuffer[pos] = ele.data
          pos += 1
    # Padding the rest
    self.__streamBuffer[pos:] = 0
    return True 

  def __cut_frame(self,streamPIPE):
    '''
    The core thread funtion to cut frames.
    '''
    print(f"{self.name}: Start...")
    try:
      while True:
        # prepare a frame of stream
        if not self.__prepare_frame_stream(streamPIPE):
          break
        # If can not add
        if self.is_wrong() or \
           self.outPIPE.is_wrong() or \
           self.outPIPE.is_terminated():
          streamPIPE.kill()
          self.kill()
          break
        # 
        else:
          ## If there are new data generated
          if (self.__firstStep and self.__tailIndex > 0) or \
             (self.__tailIndex > self.__cover):
            self.outPIPE.put( Vector(self.__streamBuffer.copy()) )
          ## check whether arrived endpoint
          if self.__endpointStep:
            self.outPIPE.put( ENDPOINT )
            self.__reset_position_flag()
          ## check whether end
          if self.__finalStep or self.is_terminated():
            self.stop()
            break
    except Exception as e:
      streamPIPE.kill()
      self.kill()
      raise e
    finally:
      print(f"{self.name}: Stop!")
  
  def _start(self,inPIPE):
    '''<inPIPE> should be a element PIPE or vector PIPE'''
    cutThread = KillableThread(target=self.__cut_frame,args=(inPIPE,))
    cutThread.setDaemon(True)
    cutThread.start()
    return cutThread

class VectorVADetector(Component):
  '''
  Do voice activity detection from a Element PIPE (expected: Audio Stream).
  We will discard the audio detected as long time silence.
  '''
  def __init__(self,frameDim,batchSize=100,patience=20,truncate=False,name=None):
    super().__init__(name=name)
    assert isinstance(frameDim,int) and frameDim > 0
    assert isinstance(batchSize,int) and batchSize > 0
    assert isinstance(patience,int) and patience > 0
    # batch size
    self.__batchSize = batchSize
    # work place
    self.__frameBuffer = np.zeros([batchSize,frameDim],dtype="int16")
    # detect function
    self.vad_function = None
    self.__silenceCounter = 0
    self.__reset_position_flag()
    #
    assert isinstance(truncate,bool)
    self.__truncate = truncate
    self.__patience = patience

  def reset(self):
    super().reset()
    self.__frameBuffer *= 0
    self.__reset_position_flag()
    self.__silenceCounter = 0

  def __reset_position_flag(self):
    self.__endpointStep = False
    self.__finalStep = False
    self.__tailIndex = self.__batchSize

  def __prepare_chunk_frame(self,framePIPE):
    '''Prepare a chunk stream data'''
    timecost = 0
    pos = 0
    while pos < self.__batchSize:
      if framePIPE.is_wrong():
        self.kill()
        return False
      elif framePIPE.is_exhausted():
        self.__tailIndex = pos
        self.__finalStep = True
        break
      # If need wait because of receiving no data
      elif framePIPE.is_empty():
        time.sleep(info.TIMESCALE)
        timecost += info.TIMESCALE
        if timecost > info.TIMEOUT:
          print(f"{self.name}: Timeout! Did not receive any data for a long time！")
          # Try to kill frame PIPE
          framePIPE.kill()
          # Kill self 
          self.kill()
          return False
      # If need wait because of blocked
      elif framePIPE.is_blocked():
        time.sleep(info.TIMESCALE)
      # If had data
      else:
        vec = framePIPE.get()
        if is_endpoint(vec):
          self.__endpointStep = True
          self.__tailIndex = pos
          break
        else:
          assert isinstance(vec,Vector), f"{self.name}: Need vector packet but got: {type(vec).__name__}."
          self.__frameBuffer[pos,:] = vec.data
          pos += 1

    # padding the tail with zero    
    self.__frameBuffer[pos:,:] = 0
    
    return True

  def __detect(self,framePIPE):
    print(f"{self.name}: Start...")
    try:
      while True:
        # prepare a chunk of frames
        self.__frameBuffer.flags.writeable = True
        if not self.__prepare_chunk_frame(framePIPE):
          break
        self.__frameBuffer.flags.writeable = False
        # Detect if necessary
        # activity can be a bool value or a list of bool value
        if self.__tailIndex > 0:
          activity = self.vad_function(self.__frameBuffer[:self.__tailIndex,:])
        else:
          activity = True
        # print(activity)
        # append data into pipe and do some processes
        if self.is_wrong() or \
           self.outPIPE.is_wrong() or \
           self.outPIPE.is_terminated():
          framePIPE.kill()
          self.kill()
          break
        else:
          #"""
          ## If this is a bool value
          if isinstance(activity,(bool,int)):
            ### If activity, add all frames in to new PIPE
            if activity:
              for i in range(self.__tailIndex):
                self.outPIPE.put( Vector( self.__frameBuffer[i].copy() ) )
              self.__silenceCounter = 0
            ### If not
            else:
              self.__silenceCounter += 1
              if self.__silenceCounter < self.__patience:
                for i in range(self.__tailIndex):
                  self.outPIPE.put( Vector( self.__frameBuffer[i].copy() ) )
              elif (self.__silenceCounter == self.__patience) and self.__truncate:
                self.outPIPE.put( ENDPOINT )
              else:
                pass
          ## if this is a list or tuple of bool value
          elif isinstance(activity,(list,tuple)):
            assert len(activity) == self.__tailIndex, f"{self.name}: If VAD detector return mutiple results, " + \
                                                      "it must has the same numbers with chunk frames."
            for i, act in enumerate(activity):
              if act:
                self.outPIPE.put( Vector( self.__frameBuffer[i].copy() ) )
                self.__silenceCounter = 0
              else:
                self.__silenceCounter += 1
                if self.__silenceCounter < self.__patience:
                  self.outPIPE.put( Vector( self.__frameBuffer[i].copy() ) )
                elif (self.__silenceCounter == self.__patience) and self.__truncate:
                  self.outPIPE.put( ENDPOINT )
          else:
            raise Exception(f"{self.name}: VAD function must return a bool value or a list of bool value.")
          #"""
          #for i in range(self.__tailIndex):
          #  self.outPIPE.put( Vector(self.__frameBuffer[i].copy()) )
          # If arrived endpoint
          if self.__endpointStep:
            self.outPIPE.put( ENDPOINT )
            self.__reset_position_flag()
          # If over
          if self.__finalStep or self.is_terminated():
            self.stop()
            break
    except Exception as e:
      framePIPE.kill()
      self.kill()
      raise e

    finally:
      print(f"{self.name}: Stop!")

  def _start(self,inPIPE):
    '''<inPIPE> should be a element PIPE'''
    assert self.vad_function is not None, f"{self.name}: Please implement the vad function."

    detectThread = KillableThread(target=self.__detect,args=(inPIPE,))
    detectThread.setDaemon(True)
    detectThread.start()

    return detectThread
