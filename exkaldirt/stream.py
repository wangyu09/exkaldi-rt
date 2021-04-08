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

#from exkaldirt.base import ExKaldiRTBase, Component, PIPE, Tunnel, Packet
#from exkaldirt.utils import run_exkaldi_shell_command
#from exkaldirt.base import info
#from exkaldirt.base import ENDPOINT, is_endpoint, NullPIPE

from base import ExKaldiRTBase, Component, PIPE, Tunnel, Packet
from utils import run_exkaldi_shell_command
from base import info, mark, print_, main_process_function
from base import ENDPOINT, is_endpoint, NullPIPE

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
  def __init__(self,waveFile,chunkSize=480,simulate=True,vaDetector=None,oKey="stream",name=None):
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

  # Process Function: Read stream
  @main_process_function
  def __read_stream(self):
    '''
    The thread function to record stream from microphone.
    '''
    readTimes = math.ceil(self.__totalframes/self.__points)
    wf = wave.open(self.__recource, "rb")

    print_(f"{self.name}: Start...")
    
    try:
      i = 0
      while i < readTimes:
        # Decide state
        master, state = self.decide_state()
        # If state is silent (although unlikely) 
        if state == mark.silent:
          self.inPIPE.activate()
          self.outPIPE.activate()
        elif state == mark.active:
          pass
        elif state in [mark.wrong,mark.terminated]:
          break
        else:
          time.sleep( info.TIMESCALE )
          continue
        #
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
            did = self.id_count
            self.put_packet( Packet( items={self.oKey:ele},sid=did,eid=did,idmaker=self.objid ) )
        elif valid is None:
          self.put_packet( ENDPOINT )

        ## if reader has been stopped by force
        if state == mark.terminated:
          self.outPIPE.stop()
          break

        # wait if necessary
        if self.__simulate:
          internal = self.__timeSpan - round( (time.time()-st),4)
          if internal > 0:
            time.sleep( internal )
        
        i += 1
        #print("Here:",i)

    except Exception as e:
      self.inPIPE.kill()
      self.outPIPE.kill()
      raise e

    else:
      if self.inPIPE.state_is_(mark.active):
        self.inPIPE.stop()
        self.outPIPE.stop()

    finally:
      wf.close()
      print_(f"{self.name}: Stop!")

  def _start(self):
    readProcess = self.create_process(target=self.__read_stream)
    readProcess.start()
    return readProcess

  def link(self,inPIPE=None,iKey=None):
    if inPIPE is None:
      inPIPE = NullPIPE()
    super().link( inPIPE=inPIPE )

  def start(self,inPIPE=None,iKey=None):
    if inPIPE is None:
      inPIPE = NullPIPE()
    super().start( inPIPE=inPIPE )

class ElementFrameCutter(Component):
  '''
  Cut frame from Element PIPE.
  '''
  def __init__(self,width=400,shift=160,oKey="frame",name=None):
    '''
    Args:
      _width_: (int) The width of sliding window.
      _shift_: (int) The shift width of each sliding.
      _name_: (str) Name.
    '''
    super().__init__(oKey=oKey,name=name)
    # Config some size parameters
    assert isinstance(width,int) and isinstance(shift,int)
    assert 0 < shift <= width
    self.__width = width
    self.__shift = shift
    self.__cover = width - shift

  def get_window_info(self):
    '''
    Get the window information.
    '''
    return namedtuple("WindowInfo",["width","shift"])(
                            self.__width,self.__shift)

  def _start(self):
    cutProcess = self.create_process(target=self.__cut_frame)
    cutProcess.start()
    return cutProcess

  ######################################
  # These functions are working on subprocess scope
  ######################################

  @main_process_function
  def __cut_frame(self):
    '''
    The core thread funtion to cut frames.
    '''
    self.__reset_position_flag()

    # Prepare a work buffer (It might be avaliable only in this process)
    self.__streamBuffer = None #np.zeros([self.__width,],dtype="int16")

    print(f"{self.name}: Start...")
    try:
      while True:
        # prepare a frame of stream
        if not self.__prepare_frame_stream():
          break
        
        ## If there are new data generated
        if (self.__firstStep and self.__tailIndex > 0) or \
            (self.__tailIndex > self.__cover):
          eid = self.id_count
          self.put_packet( Packet( items={self.oKey:self.__streamBuffer.copy()},sid=eid,eid=eid,idmaker=self.objid ) )
        ## check whether arrived endpoint
        if self.__endpointStep:
          self.put_packet( ENDPOINT )
          self.__reset_position_flag()
        ## check whether end
        if self.__finalStep:
          self.outPIPE.stop()
          break

    except Exception as e:
      self.inPIPE.kill()
      self.outPIPE.kill()
      raise e

    finally:
      print(f"{self.name}: Stop!")

  def __reset_position_flag(self):
    '''
    Some flags to mark position of indexes.
    '''
    self.__zerothStep = True
    self.__firstStep = False
    self.__endpointStep = False
    self.__finalStep = False
    self.__tailIndex = self.__width

  def __prepare_frame_stream(self):
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
      # Decide state
      master, state = self.decide_state()

      # If state is silent (although unlikely) 
      if state == mark.silent:
        self.inPIPE.activate()
        self.outPIPE.activate()
      elif state == mark.active:
        pass   
      elif state == mark.wrong:
        return False
      elif state == mark.terminated:
        if master == mark.outPIPE:
          return False
        else:
          if self.inPIPE.is_empty():
            self.__finalStep = True
            self.__tailIndex = pos
            break
      else:
        time.sleep( info.TIMESCALE )
        continue
      # 
      if self.inPIPE.is_empty():
        time.sleep(info.TIMESCALE)
        timecost += info.TIMESCALE
        if timecost > info.TIMEOUT:
          print(f"{self.name}: Timeout!")
          self.inPIPE.kill()
          self.outPIPE.kill()
          return False
        else:
          continue
      else:
        pack = self.get_packet()
        if is_endpoint(pack):
          self.__endpointStep = True
          self.__tailIndex = pos
          break
        else:
          iKey = pack.mainKey if self.iKey is None else self.iKey
          ele = pack[ iKey ]
          assert isinstance(ele, (np.signedinteger,np.floating))
          if self.__streamBuffer is None:
            self.__streamBuffer = np.zeros([self.__width,], dtype=ele.dtype)
          self.__streamBuffer[pos] = ele
          pos += 1

    # Padding the rest
    if self.__streamBuffer is not None:
      self.__streamBuffer[pos:] = 0
    return True


  
