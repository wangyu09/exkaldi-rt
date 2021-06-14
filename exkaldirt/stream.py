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
import multiprocessing
import numpy as np
import subprocess
from collections import namedtuple

from exkaldirt.base import ExKaldiRTBase, Component, PIPE, Packet, ContextManager
from exkaldirt.utils import run_exkaldi_shell_command
from exkaldirt.base import info, mark, print_
from exkaldirt.base import Endpoint, is_endpoint, NullPIPE

# from base import ExKaldiRTBase, Component, PIPE, Packet, ContextManager
# from utils import run_exkaldi_shell_command
# from base import info, mark, print_
# from base import Endpoint, is_endpoint, NullPIPE

def record(seconds=0,fileName=None):
  '''
  Record audio stream from microphone.
  The format is restricted: sampling rate -> 16KHz, data type -> int16, channels -> 1.

  Args:
    _seconds_: (int) If == 0, do not limit time.
    _fileName_: (str) If None, return wave data.
  
  Return:
    A string: If _fileName_ is a path.
    A Wave object: hold audio information and data if _fileName_ is None. 
  '''
  assert isinstance(seconds,(int,float)) and seconds >= 0

  rate = 16000
  width = 2
  channels = 1
  paFormat = pyaudio.paInt16
  npFormat = "int16"
  perCount = 1600

  totCount = int(seconds*rate)
  if totCount != 0 and totCount < perCount:
    raise Exception("Recording time is extremely short!")
  recordTimes = int(totCount/perCount)

  pa = pyaudio.PyAudio()
  stream = pa.open(format=paFormat,channels=channels,rate=rate,
              input=True,output=False)
  result = []
  try:
    if seconds == 0:
      print("Start recording...")
      while True:
        result.append(stream.read(perCount))
    else:
      print("Start recording...")
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

def read(waveFile):
  '''
  Read audio data from wave file.

  Args:
    _waveFile_: (str) wave file path.
  '''
  assert os.path.isfile(waveFile), f"No such file: {waveFile}"

  with wave.open(waveFile, "rb") as wf:
    rate = wf.getframerate()
    channels = wf.getnchannels()
    width = wf.getsampwidth()
    frames = wf.getnframes()
    data = np.frombuffer(wf.readframes(frames),dtype="int"+str(8*width))

  duration = round(frames/rate,4)
  if channels > 1:
    data = data.reshape([-1,channels])

  return namedtuple("Wave",["rate","channels","points","duration","value"])(
                    rate,channels,frames,duration,data
                  )

def write(waveform,fileName,rate=16000,channels=1):
  '''
  Write waveform to file.
  '''
  assert isinstance(waveform,np.ndarray) and waveform.dtype == 'int', \
                        "_waveform_ should be an int array."
  assert waveform.dtype == "int"
  width = waveform.dtype.alignment
  assert isinstance(rate,int) and rate > 0
  assert isinstance(channels,int) and channels > 0
  assert isinstance(fileName,str)
  # Some preparation
  fileName = fileName.strip()
  if not fileName.endswith(".wav"):
    fileName += ".wav"
  dirName = os.path.dirname(fileName)
  if not os.path.isdir(dirName):
    os.makedirs(dirName)
  if os.path.isfile(fileName):
    os.remove(fileName)
  # Write
  with wave.open(fileName,'wb') as wf:
    wf.setnchannels(channels) 
    wf.setsampwidth(width) 
    wf.setframerate(rate) 
    wf.writeframes( waveform.tobytes() ) 
  # Return actual file name
  return fileName

def cut_frames(waveform,width=400,shift=160,snip=True):
  '''
  Cut a wave data into N frames.
  The rest will be discarded.

  Args:
    _waveform_: A 1-d NumPy array.
    _width_: An int value. The sliding window width.
    _shift_: An int value. The step width. Must <= _width_:
  
  Return:
    A 2-d array.
  '''
  assert isinstance(waveform,np.ndarray) and len(waveform.shape) == 1, "<waveform> must be a 1-d NumPy array."
  assert isinstance(width,int) and width > 0
  assert isinstance(shift,int) and 0 < shift <= width
  assert isinstance(snip,bool)
  points = len(waveform)
  assert points >= width
  
  N = int((points-width)/shift) + 1

  if ( N * shift + (width - shift) ) < points and not snip:
    N += 1

  result = np.zeros([N,width],dtype=waveform.dtype)
  for i in range(N):
    offset = i * shift
    rest = width if offset + width <= points else points - offset
    result[i,0:rest] = waveform[ offset:offset+rest ]

  return result

class VADetector(ExKaldiRTBase):
  '''
  Voice Activity Detector used to be embeded in StreamReader or StreamRecorder.
  Note that this is not a Component object.
  '''
  def __init__(self,patience=7,truncate=False,name=None):
    '''
    Args:
      _patience_: (int) The maximum length of continuous silence.
      _truncate_: (bool) If True, truncate the stream if the length of continuous silence >= _patience_.
      _name_: (str) The name of this detector.
    '''
    # Initialize state and name
    super().__init__(name=name)
    # The maximum continuous silence length.
    assert isinstance(patience,int) and patience > 0
    self.__patience = patience
    #
    assert isinstance(truncate,bool)
    self.__truncate = truncate
    # Config others
    self.__silenceCounter = 0
    # Core function
    self.is_speech = None

  def reset(self):
    '''
    Reset the detector.
    '''
    self.__silenceCounter = 0

  def detect(self,chunk):
    '''
    Detect whether a chunk of stream is speech or silence.

    Args:
      _chunk_: (bytes) a chunk stream.

    Return:
      _True_: This chunk data should be retained.
      _False_: This chunk data should be discarded.
      _None_: This chunk data should be replaced with ENDPOINT.
    '''
    assert callable(self.is_speech), f"{self.name}: Please implement .is_speech function."
    assert isinstance(chunk,bytes)
    activity = self.is_speech(chunk)
    assert isinstance(activity,bool)
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
  '''
  A detector based on Google Webrtc VAD
  Note that this is not a Component object.
  If you embed this detector into StreamReader or StreamRecoder.
  Note that the size of chunk stream is restricted.
  '''
  def __init__(self,patience=7,mode=3,truncate=False,name=None):
    '''
    Args:
      _patience_: (int) The maximum length of continuous endpoints.
      _truncate_: (bool) If True, truncate the stream if the length of continuous endpoints >= _patience_.
      _mode_: (int), 1-3, the mode of webrtcvad object.
      _name_: (str) The name of this detector.
    '''
    # Initialize
    super().__init__(patience=patience,truncate=truncate,name=name)
    # Define a webrtcvad object.
    assert mode in [1,2,3], f"{self.name}: <mode> must be 1, 2 or 3, but got: {mode}."
    self.__webrtcvadobj = webrtcvad.Vad(mode)
    self.is_speech = self.__is_speech

  def __is_speech(self,chunk)->bool:
    return self.__webrtcvadobj.is_speech(chunk,16000)

class StreamReader(Component):
  '''
  Record real-time audio stream from wave file.
  In current version, the format is restricted: [ sampling rate = 16KHz, data type = int16, channels = 1 ].
  '''
  def __init__(self,waveFile,chunkSize=480,simulate=True,vaDetector=None,oKey="data",name=None):
    '''
    Args:
      _waveFile_: (str) A wave file path.
      _chunkSize_: (int) How many sampling points of each reading. 
      _simulate_: (bool) If True, simulate actual clock.
      _vaDetector_: (VADetector) A VADetector object to detect the stream.
      _name_: (str) Name.
    '''
    super().__init__(oKey=oKey,name=name)
    # Config audio information
    self.__audio_info = self.__direct_source(waveFile)
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
    # A flag to set sampling id
    self.__id_counter = 0
    #
    self.link( NullPIPE() )
  
  @property
  def __id_count(self):
    self.__id_counter += 1
    return self.__id_counter - 1

  def __direct_source(self,waveFile):
    '''
    Get audio information from a wave file.
    '''
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

  def __config_format(self,Rate,Channels,Width,Frames,Duration):
    '''
    Set wave parameters.
    '''
    assert Width in [2,4]

    self.__rate = Rate
    self.__channels = Channels
    self.__width = Width
    self.__format = f"int{ Width * 8 }"
    
    return namedtuple("AudioInfo",["rate","channels","width","frames","duration"])(
                                    Rate,Channels,Width,Frames,Duration)

  def get_audio_info(self):
    '''
    Return the audio information.
    '''
    return self.__audio_info 

  def redirect(self,waveFile):
    '''
    Bind a new wave file.

    Args:
      _waveFile_: (str) A wave file path.
    '''
    if not self.outPIPE.state_is_(mark.active,mark.stranded):
      raise Exception( f"{self.name}: Stream Reader is running. Can not redirect.")
    
    # Reset this component firstly
    self.reset()
    self.__audio_info = self.__direct_source(waveFile)

  def link(self,inPIPE=None,iKey=None):
    if inPIPE is None:
      inPIPE = NullPIPE()
    super().link( inPIPE=inPIPE )

  def start(self,inPIPE=None,iKey=None):
    if inPIPE is None and self.inPIPE is None:
      inPIPE = NullPIPE()
    super().start( inPIPE=inPIPE )
  
  def core_loop(self):
    '''
    The thread function to record stream from microphone.
    '''
    readTimes = math.ceil(self.__totalframes/self.__points)
    wf = wave.open(self.__recource, "rb")

    try:
      i = 0
      while i < readTimes:
        # Decide state
        master, state = self.decide_state()
        #print("master:",master,"state:",state,"inPIPE state:",self.inPIPE.state,"outPIPT state:",self.outPIPE.state)
        # If state is silent (although unlikely) 
        if state in [mark.wrong,mark.terminated]:
          break
        elif state == mark.stranded:
          time.sleep( info.TIMESCALE )
          if self.__redirect_flag:
            break
          continue
        #
        #print( "try to read stream" )
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
          ## append data
          for ele in np.frombuffer(data,dtype=self.__format):
            if self.outPIPE.state_is_(mark.silent,mark.active):
              self.put_packet( Packet( items={self.oKey[0]:ele},cid=self.__id_count,idmaker=self.objid ) )
        elif valid is None:
          self.put_packet( Endpoint( cid=self.__id_count,idmaker=self.objid ) )
        ## if reader has been stopped by force
        if state == mark.terminated:
          break
        #print( "sleep" )
        # wait if necessary
        if self.__simulate:
          internal = self.__timeSpan - round( (time.time()-st),4)
          if internal > 0:
            time.sleep( internal )
        
        i += 1
    finally:
      wf.close()

class StreamRecorder(Component):
  '''
  Record real-time audio stream from wave file.
  In current version, the format is restricted: [ sampling rate = 16KHz, data type = int16, channels = 1 ].
  '''
  def __init__(self,chunkSize=480,vaDetector=None,oKey="data",name=None):
    '''
    Args:
      _waveFile_: (str) A wave file path.
      _chunkSize_: (int) How many sampling points of each reading. 
      _simulate_: (bool) If True, simulate actual clock.
      _vaDetector_: (VADetector) A VADetector object to detect the stream.
      _name_: (str) Name.
    '''
    super().__init__(oKey=oKey,name=name)
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
    # Config audio information
    self.__audio_info = self.__config_format(Rate=16000,Channels=1,Width=2)
    # A flag to set sampling id
    self.__id_counter = 0
    #
    self.link( NullPIPE() )
  
  @property
  def __id_count(self):
    self.__id_counter += 1
    return self.__id_counter - 1

  def __config_format(self,Rate,Channels,Width):
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
    
    return namedtuple("AudioInfo",["rate","channels","width",])(
                                    Rate,Channels,Width)
  
  def get_audio_info(self):
    '''
    Return the audio information.
    '''
    return self.__audio_info 

  def link(self,inPIPE=None,iKey=None):
    if inPIPE is None:
      inPIPE = NullPIPE()
    super().link( inPIPE=inPIPE )

  def start(self,inPIPE=None,iKey=None):
    if inPIPE is None and self.inPIPE is None:
      inPIPE = NullPIPE()
    super().start( inPIPE=inPIPE )
  
  def core_loop(self):
    '''
    The thread function to record stream from microphone.
    '''
    pa = pyaudio.PyAudio()
    stream = pa.open(format=self.__paFormat,channels=self.__channels,
                     rate=self.__rate,input=True,output=False)
    try:
      while True:
        # 
        master, state = self.decide_state()
        #
        if state in [mark.wrong,mark.terminated]:
          break
        elif state == mark.stranded:
          time.sleep( info.TIMESCALE )
          if self.__redirect_flag:
            break
          continue
        
        data = stream.read(self.__points)
        # detcet if necessary
        if self.__vad is not None:
          valid = self.__vad.detect(data)
        else:
          valid = True
        # add data
        if valid is True:
          ## append data
          for ele in np.frombuffer(data,dtype=self.__format):
            if self.outPIPE.state_is_(mark.silent,mark.active):
              self.put_packet( Packet( items={self.oKey[0]:ele},cid=self.__id_count,idmaker=self.objid ) )
        elif valid is None:
          self.put_packet( Endpoint( cid=self.__id_count,idmaker=self.objid ) )

        ## if reader has been stopped by force
        if state == mark.terminated:
          break
      
    finally:
      stream.stop_stream()
      stream.close()
      pa.terminate()

class ElementFrameCutter(Component):
  '''
  Cut frame from Element PIPE.
  '''
  def __init__(self,batchSize=50,width=400,shift=160,oKey="data",name=None):
    '''
    Args:
      _batchSize_: (int) Batch size. If > 1, output matrix, else, output vector.
      _width_: (int) The width of sliding window.
      _shift_: (int) The shift width of each sliding.
      _name_: (str) Name of component.
    '''
    super().__init__(oKey=oKey,name=name)
    # Config some size parameters. 
    assert isinstance(width,int) and isinstance(shift,int)
    assert 0 < shift <= width
    assert batchSize >= 1 

    self.__width = width
    self.__shift = shift
    self.__cover = width - shift
    self.__batchSize = batchSize

    self.__id_counter = 0
  
  @property
  def __id_count(self):
    self.__id_counter += 1
    return self.__id_counter - 1

  def get_window_info(self):
    '''
    Get the window information.
    '''
    return namedtuple("WindowInfo",["width","shift"])(
                            self.__width,self.__shift)

  ######################################
  # These functions are working on subprocess scope
  ######################################

  def core_loop(self):
    '''
    The core thread funtion to cut frames.
    '''
    self.__reset_position_flag()
    # Prepare a work buffer (It might be avaliable only in this process)
    self.__streamBuffer = None #np.zeros([self.__width,],dtype="int16")

    while True:
      # prepare a frame of stream
      if not self.__prepare_chunk_stream():
        break
      ## If there are new data generated
      if self.__hadData:
        if self.__batchSize == 1:
          self.put_packet( Packet( items={self.oKey[0]:self.__streamBuffer[0].copy()}, cid=self.__id_count, idmaker=self.objid ) )
        else:
          self.put_packet( Packet( items={self.oKey[0]:self.__streamBuffer.copy()}, cid=self.__id_count, idmaker=self.objid ) )
      ## check whether arrived endpoint
      if self.__endpointStep:
        self.put_packet( Endpoint( cid=self.__id_count,idmaker=self.objid ) )
        self.__reset_position_flag()
      ## check whether end
      if self.__finalStep:
        break

  def __reset_position_flag(self):
    '''
    Some flags to mark position of indexes.
    '''
    self.__zerothStep = True
    self.__endpointStep = False
    self.__finalStep = False
    self.__hadData = False

  def __prepare_chunk_stream(self):
    '''
    Prepare chunk stream to compute feature.
    '''
    self.__hadData = False

    for i in range(self.__batchSize):

      # copy old data if necessary
      if self.__zerothStep:
        pos = 0
        self.__zerothStep = False
      else:
        self.__streamBuffer[i,0:self.__cover] = self.__streamBuffer[i-1,self.__shift:]
        pos = self.__cover

      # get new data
      while pos < self.__width:
        # Decide action
        action = self.decide_action()
        # 
        if action is True:
          pack = self.get_packet()
          if not pack.is_empty():
            iKey = pack.mainKey if self.iKey is None else self.iKey
            ele = pack[ iKey ]
            assert isinstance(ele, (np.signedinteger,np.floating))
            if self.__streamBuffer is None:
              self.__streamBuffer = np.zeros([self.__batchSize,self.__width,], dtype=ele.dtype)
            self.__streamBuffer[i,pos] = ele
            self.__hadData = True
            pos += 1
          if is_endpoint(pack):    
            self.__endpointStep = True
            break
        elif action is None:
          self.__finalStep = True
          break
        else:
          return False

      # Padding the rest
      if self.__streamBuffer is not None:
        self.__streamBuffer[i,pos:] = 0
      
      if self.__endpointStep or self.__finalStep:
        break

    if self.__streamBuffer is not None:
      self.__streamBuffer[i+1:] = 0

    return True

class FrameDissolver(Component):

  def __init__(self,oKey="data",name=None):
    super().__init__(oKey=oKey,name=name)
    self.__id_counter = 0
  
  @property
  def __id_count(self):
    self.__id_counter += 1
    return self.__id_counter - 1
  
  def core_loop(self):
    while True:

      action = self.decide_action()
      if action is True:
        packet = self.get_packet()
        if not packet.is_empty():
          iKey = self.iKey if self.iKey is not None else packet.mainKey
          data = packet[iKey]
          assert isinstance(data,np.ndarray), f"{self.name}: Can only dissolve vector and matrix packet but got: {type(data)}."
          for element in data.reshape(-1):
            self.put_packet( Packet( {self.oKey[0]:element},cid=self.__id_count,idmaker=packet.idmaker ) )
        if is_endpoint(packet):
          self.put_packet( Endpoint(cid=self.__id_count,idmaker=packet.idmaker) )
      else:
        break

class VectorBatcher(Component):

  def __init__(self,center,left=0,right=0,oKey="data",name=None):
    super().__init__(oKey=oKey,name=name)
    assert isinstance(center,int) and center > 0
    self.__center = center
    self.__left = left
    self.__right = right
    self.__width = center + left + right
    self.__cover = left + right
    self.__id_counter = 0

  @property
  def __id_count(self):
    self.__id_counter += 1
    return self.__id_counter - 1
  
  def get_batch_info(self):
    '''
    Get the window information.
    '''
    return namedtuple("BatchInfo",["center","left","right"])(
                            self.__center,self.__left,self.__right)

  def core_loop(self):
    '''
    The core thread funtion to batch.
    '''
    self.__reset_position_flag()
    # Prepare a work buffer (It might be avaliable only in this process)
    self.__streamBuffer = None #np.zeros([self.__width,],dtype="int16")

    while True:
      # prepare a frame of stream
      if not self.__prepare_batch_stream():
        break
      ## If there are new data generated
      if self.__hadData:
        self.put_packet( Packet( items={self.oKey[0]:self.__streamBuffer.copy()}, cid=self.__id_count, idmaker=self.objid ) )
      ## check whether arrived endpoint
      if self.__endpointStep:
        self.put_packet( Endpoint( cid=self.__id_count,idmaker=self.objid ) )
        self.__reset_position_flag()
      ## check whether end
      if self.__finalStep:
        break

  def __reset_position_flag(self):
    '''
    Some flags to mark position of indexes.
    '''
    self.__zerothStep = True
    self.__firstStep = False
    self.__endpointStep = False
    self.__finalStep = False
    self.__hadData = False

  def __prepare_batch_stream(self):
    '''
    Prepare chunk stream to compute feature.
    '''
    self.__hadData = False

    # copy old data if necessary
    if self.__zerothStep:
      pos = self.__left
      self.__zerothStep = False
      self.__firstStep = True
    else:
      self.__streamBuffer[0:self.__cover] = self.__streamBuffer[self.__center:]
      pos = self.__cover
      self.__firstStep = False

    # get new data
    while pos < self.__width:
      # Decide state
      action = self.decide_action()
      if action is True:
        pack = self.get_packet()
        if not pack.is_empty():
          iKey = pack.mainKey if self.iKey is None else self.iKey
          vec = pack[ iKey ]
          assert isinstance(vec, np.ndarray) and len(vec.shape) == 1
          if self.__streamBuffer is None:
            dim = len(vec)
            self.__streamBuffer = np.zeros([self.__width,dim,], dtype=vec.dtype)
          self.__streamBuffer[pos] = vec
          self.__hadData = True
          pos += 1
        if is_endpoint(pack):
          self.__endpointStep = True
          break
      elif action is False:
        return False
      else:
        self.__finalStep = True
        break

    # Padding the rest
    if self.__streamBuffer is not None:
      self.__streamBuffer[pos:] = 0

    return True   

class MatrixSubsetter(Component):

  def __init__(self,nChunk=2,oKey="data",name=None):
    super().__init__(oKey=oKey,name=name)
    assert isinstance(nChunk,int) and nChunk > 1
    self.__nChunk = nChunk  
  
    self.__id_counter = 0

  @property
  def __id_count(self):
    self.__id_counter += 1
    return self.__id_counter - 1

  def core_loop(self):
    '''
    The core thread funtion to batch.
    '''
    while True:
      # Decide action
      action = self.decide_action()
      if action is True:
        # get a packet
        pack = self.get_packet()
        if not pack.is_empty():
          iKey = self.iKey if self.iKey is not None else pack.mainKey
          mat = pack[iKey]
          assert isinstance(mat,np.ndarray) and len(mat.shape) == 2
          cSize = len(mat) // self.__nChunk
          assert cSize * self.__nChunk == len(mat)
          # Split matrix
          for i in range(self.__nChunk):
            self.put_packet( Packet(items={self.oKey[0]:mat[i*cSize:(i+1)*cSize]}, cid=self.__id_count, idmaker=pack.idmaker) )
        # add endpoint
        if is_endpoint(pack):
          self.put_packet( Endpoint(cid=self.__id_count, idmaker=pack.idmaker) )
      else:
        break

class VectorVADetector(Component):
  '''
  Do voice activity detection from a Element PIPE (expected: Audio Stream).
  We will discard the audio detected as long time silence.
  '''
  def __init__(self,batchSize,vadFunc,patience=20,truncate=False,oKey="data",name=None):
    '''
    Args:
      _frameDim_: (int) The dims of vector.
      _batchSize_: (int) Batch size.
      _patience_: (int) The maximum length of continuous endpoints.
      _truncate_: (bool) If True, truncate the stream if the length of continuous endpoints >= _patience_.
      _name_: (str) Name.
    '''
    super().__init__(oKey=oKey,name=name)
    # batch size
    assert isinstance(batchSize,int) and batchSize > 0
    self.__batchSize = batchSize
    # detect function
    assert callable(vadFunc)
    self.vad_function = vadFunc
    self.__silenceCounter = 0
    self.__reset_position_flag()
    #
    assert isinstance(truncate,bool)
    self.__truncate = truncate
    assert isinstance(patience,int) and patience > 0
    self.__patience = patience

    self.__id_counter = 0
  
  @property
  def __id_count(self):
    self.__id_counter += 1
    return self.__id_counter - 1

  def reset(self):
    '''
    Reset.
    '''
    super().reset()
    self.__reset_position_flag()
    self.__silenceCounter = 0

  def __reset_position_flag(self):
    self.__endpointStep = False
    self.__finalStep = False
    self.__tailIndex = self.__batchSize

  def core_loop(self):

    self.__reset_position_flag()
    # Prepare a work buffer (It might be avaliable only in this process)
    self.__workBuffer = None #np.zeros([self.__width,],dtype="int16")

    while True:
      # prepare a chunk of frames
      if not self.__prepare_chunk_frame():
        break
      self.__workBuffer.flags.writeable = False
      # Detect if necessary
      # activity can be a bool value or a list of bool value
      if self.__tailIndex > 0:
        activity = self.vad_function( self.__workBuffer[:self.__tailIndex] )
      else:
        activity = True
      self.__workBuffer.flags.writeable = True
      # print(activity)
      # append data into pipe and do some processes

      if isinstance(activity,(bool,int)):
        ### If activity, add all frames in to new PIPE
        if activity:
          for i in range(self.__tailIndex):
            self.put_packet( Packet({self.oKey[0]:self.__workBuffer[i].copy()},cid=self.__id_count,idmaker=self.objid) )
          self.__silenceCounter = 0
        ### If not
        else:
          self.__silenceCounter += 1
          if self.__silenceCounter < self.__patience:
            for i in range(self.__tailIndex):
              self.put_packet( Packet({self.oKey[0]:self.__workBuffer[i].copy()},cid=self.__id_count,idmaker=self.objid) )
          elif (self.__silenceCounter == self.__patience) and self.__truncate:
            self.put_packet( Endpoint(cid=self.__id_count,idmaker=self.objid) )
          else:
            pass
      ## if this is a list or tuple of bool value
      elif isinstance(activity,(list,tuple)):
        assert len(activity) == self.__tailIndex, f"{self.name}: If VAD detector return mutiple results, " + \
                                                  "it must has the same numbers with chunk frames."
        for i, act in enumerate(activity):
          if act:
            self.put_packet( Packet({self.oKey[0]:self.__workBuffer[i].copy()},cid=self.__id_count,idmaker=self.objid) )
            self.__silenceCounter = 0
          else:
            self.__silenceCounter += 1
            if self.__silenceCounter < self.__patience:
              self.put_packet( Packet({self.oKey[0]:self.__workBuffer[i].copy()},cid=self.__id_count,idmaker=self.objid) )
            elif (self.__silenceCounter == self.__patience) and self.__truncate:
              self.put_packet( Endpoint(cid=self.__id_count,idmaker=self.objid) )
      else:
        raise Exception(f"{self.name}: VAD function must return a bool value or a list of bool value.")
      # If arrived endpoint
      if self.__endpointStep:
        self.put_packet( Endpoint(cid=self.__id_count,idmaker=self.objid) )
        self.__reset_position_flag()
      # If over
      if self.__finalStep:
        break

  def __prepare_chunk_frame(self):
    '''Prepare a chunk stream data'''

    pos = 0
    while pos < self.__batchSize:
      action = self.decide_action()
      if action is True:
        pack = self.get_packet()
        if not pack.is_empty():
          iKey = pack.mainKey if self.iKey is None else self.iKey
          vec = pack[ iKey ]
          assert isinstance(vec, np.ndarray) and len(vec.shape) == 1
          if self.__workBuffer is None:
            dim = len(vec)
            self.__workBuffer = np.zeros([self.__batchSize,dim,], dtype=vec.dtype)
          self.__workBuffer[pos] = vec
          pos += 1  
        if is_endpoint(pack):
          self.__endpointStep = True
          self.__tailIndex = pos
          break
      elif action is None:
        self.__finalStep = True
        self.__tailIndex = pos
        break
      else:
        return False

    # padding the tail with zero    
    self.__workBuffer[pos:] = 0
    
    return True