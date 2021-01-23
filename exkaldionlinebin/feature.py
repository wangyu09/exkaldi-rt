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

import numpy as np
import threading
import time
import os

from base import StateFlag, PIPE, Vector
from base import run_shell_command, encode_vector
from base import TIMEOUT, TIMESCALE, EPSILON, CMDROOT

def cut_frames(waveform,width,shift):
  points = len(waveform)
  assert points >= width
  
  N = int((points-width)/shift) + 1
  result = np.zeros([N,width],dtype="int16")
  for i in range(N):
    offset = i * shift
    result[i] = waveform[ offset:offset+width ]
  
  return result

def pre_emphasize_1d(waveform,coeff=0.95):
  assert 0 <= coeff < 1.0
  assert len(waveform.shape) == 1
  new = np.zeros_like(waveform)
  new[1:] = waveform[1:] - coeff*waveform[:-1]
  new[0] = waveform[0] - coeff*waveform[0]
  return new

def pre_emphasize_2d(waveform,coeff=0.95):
  assert 0 <= coeff < 1.0
  assert len(waveform.shape) == 2
  new = np.zeros_like(waveform)
  new[:,1:] = waveform[:,1:] - coeff*waveform[:,:-1]
  new[:,0] = waveform[:,0] - coeff*waveform[:,0]
  return new

def get_window_function(size,winType="povey",blackmanCoeff=0.42):
  window = np.zeros([size,],dtype="float32")
  a = 2*np.pi / (size-1)
  for i in range(size):
    if winType == "hanning":
      window[i] = 0.5 - 0.5*np.cos(a*i)
    elif winType == "sine":
      window[i] = np.sin(0.5*a*i)
    elif winType == "hamming":
      window[i] = 0.54 - 0.46*np.cos(a*i)
    elif winType == "povey":
      window[i] = (0.5-0.5*np.cos(a*i))**0.85
    elif winType == "rectangular":
      winType[i] = 1.0
    elif winType == "blackman":
      winType[i] = blackmanCoeff - 0.5*np.cos(a*i) + (0.5-blackmanCoeff)*np.cos(2*a*i)
    else:
      raise Exception(f"Unknown Window Type: {winType}")
  
  return window

def dither_singal_1d(waveform,factor=0.0):
  assert len(waveform.shape) == 1
  wavData = f"1 {len(waveform)} ".encode() + encode_vector(waveform)
  cmd = os.path.join(CMDROOT,f"exkaldi-dither --factor {factor}")
  out = run_shell_command(cmd,inputs=wavData)
  return np.array(out,dtype="float32")

def dither_singal_2d(waveform,factor=0.0):
  assert len(waveform.shape) == 2
  rows = waveform.shape[0]
  cols = waveform.shape[1]
  wavData = f"{rows} {cols} ".encode() + encode_vector(waveform.reshape(-1))
  cmd = os.path.join(CMDROOT,f"exkaldi-dither --factor {factor}")
  out = run_shell_command(cmd,inputs=wavData)
  return np.array(out,dtype="float32").reshape(rows,cols)

def remove_dc_offset_1d(waveform):
  assert len(waveform.shape) == 1
  return waveform - np.mean(waveform)

def remove_dc_offset_2d(waveform):
  assert len(waveform.shape) == 2
  return waveform - np.mean(waveform,axis=1,keepdims=True)

def compute_log_energy_1d(waveform,floor=EPSILON):
  assert len(waveform.shape) == 1
  return np.log(max(np.sum(waveform**2),floor))

def compute_log_energy_2d(waveform,floor=EPSILON):
  assert len(waveform.shape) == 2
  temp = np.sum(waveform**2,axis=1)
  temp[temp < floor] = floor
  return np.log(temp)

def split_radix_real_fft_1d(waveform):
  assert len(waveform.shape) == 1
  points = len(waveform)
  fftLen = get_padded_fft_length(points)
  inputs = f"1 {points} ".encode() + encode_vector(waveform)
  cmd = os.path.join(CMDROOT,f"exkaldi-srfft --fftsize {fftLen}")
  out = run_shell_command(cmd,inputs=inputs)
  return fftLen, np.array(out,dtype="float32").reshape([-1,2])

def split_radix_real_fft_2d(waveform):
  assert len(waveform.shape) == 2
  frames = waveform.shape[0]
  points = waveform.shape[1]
  fftLen = get_padded_fft_length(points)
  inputs = f"{frames} {points} ".encode() + encode_vector(waveform.reshape(-1))
  cmd = os.path.join(CMDROOT,f"exkaldi-srfft --fftsize {fftLen}")
  out = run_shell_command(cmd,inputs=inputs)
  return fftLen, np.array(out,dtype="float32").reshape([frames,-1,2])

def compute_power_spectrum_1d(fftFrame):

  assert len(fftFrame.shape) == 2
  
  zeroth = fftFrame[0,0] + fftFrame[0,1]
  n2th = fftFrame[0,0] - fftFrame[0,1]
  
  fftFrame = np.sum(fftFrame**2,axis=1)
  fftFrame[0] = zeroth**2

  return np.append(fftFrame,n2th**2)

def compute_power_spectrum_2d(fftFrame):

  assert len(fftFrame.shape) == 3
  
  zeroth = fftFrame[:,0,0] + fftFrame[:,0,1]
  n2th = fftFrame[:,0,0] - fftFrame[:,0,1]
  
  fftFrame = np.sum(fftFrame**2,axis=2)
  fftFrame[:,0] = zeroth**2

  return np.append(fftFrame,(n2th**2)[:,None],axis=1)

def apply_floor(feature,floor=EPSILON):
  feature[feature<floor] = floor
  return feature

def mel_scale(freq):
  return 1127.0 * np.log (1.0 + freq / 700.0)

def inverse_mel_scale(melFreq):
  return 700.0 * (np.exp(melFreq/1127.0) - 1)

def get_mel_bins(numBins,rate,fftLen,lowFreq=20,highFreq=0):
    
  nyquist = 0.5 * rate
  numFftBins = fftLen//2

  if highFreq <= 0:
    highFreq = nyquist + highFreq

  fftBinWidth = rate/fftLen
  melLow = mel_scale(lowFreq)
  melHigh = mel_scale(highFreq)

  delDelta = (melHigh-melLow)/(numBins+1)

  result = np.zeros([numBins,numFftBins+1],dtype="float32")
  for binIndex in range(numBins):
    leftMel = melLow + binIndex * delDelta
    centerMel = melLow + (binIndex+1) * delDelta
    rightMel = melLow + (binIndex+2) * delDelta
    for i in range(numFftBins):
      freq = fftBinWidth * i
      mel = mel_scale(freq)

      if leftMel < mel <  rightMel:
        if mel <= centerMel:
          weight = (mel - leftMel)/(centerMel - leftMel)
        else:
          weight = (rightMel - mel)/(rightMel - centerMel)
        result[binIndex,i] = weight

  return result.T

def get_padded_fft_length(points):
  fftLen = 1
  while fftLen < points:
    fftLen <<= 1
  return fftLen

def get_dct_matrix(numCeps,numBins):
  result = np.zeros([numCeps,numBins],dtype="float32")
  result[0] = np.sqrt(1/numBins)
  normalizer = np.sqrt(2/numBins)
  for i in range(1,numCeps):
    for j in range(0,numBins):
      result[i,j] = normalizer * np.cos( np.pi/numBins*(j+0.5)*i )
  return result.T

def get_cepstral_lifter_coeff(dim,factor=22):
  assert factor > 0
  result = np.zeros([dim,],dtype="float32")
  for i in range(dim):
    result[i] = 1.0 + 0.5*factor*np.sin(np.pi*i/factor)
  return result

def add_deltas(feat, order=2, window=2):
  assert len(feat.shape) == 2
  frames = feat.shape[0]
  dims = feat.shape[1]
  inputs = f"{frames} {dims} ".encode() + encode_vector( feat.reshape(-1) )
  cmd = os.path.join(CMDROOT,f"exkaldi-add-deltas --order {order} --window {window}")
  out = run_shell_command(cmd,inputs=inputs)
  return np.array(out,dtype="float32").reshape([frames,-1])

def splice_feats(feat, left, right):
  assert len(feat.shape) == 2
  frames = feat.shape[0]
  dims = feat.shape[1]
  inputs = f"{frames} {dims} ".encode() + encode_vector( feat.reshape(-1) )
  cmd = os.path.join(CMDROOT,f"exkaldi-splice-feats --left {left} --right {right}")
  out = run_shell_command(cmd, inputs=inputs)
  return np.array(out,dtype="float32").reshape([frames,-1])

# This function is wrapped from kaldi_io library.
def load_lda_matrix(ldaFile):
  assert os.path.isfile(ldaFile)
  with open(ldaFile,"rb") as fd:
    binary = fd.read(2).decode()
    assert binary == '\0B'
    header = fd.read(3).decode()
    if header == 'FM ':
      sample_size = 4
    elif header == 'DM ':
      sample_size = 8
    else:
      raise Exception("Only FM -> float32 or DM -> float64 can be used.")
    s1, rows, s2, cols = np.frombuffer(fd.read(10), dtype='int8,int32,int8,int32', count=1)[0]
    buf = fd.read(rows * cols * sample_size)
    if sample_size == 4 : 
      vec = np.frombuffer(buf, dtype='float32')
    else:
      vec = np.frombuffer(buf, dtype='float64')
    return np.reshape(vec,(rows,cols)).T

class FeatureExtractor(StateFlag):
  '''
  The base class of a feature extractor.
  Please implement the self.extractFcuntion by your fucntion.
  '''
  def __init__(self,frameDim=400,chunkFrames=10):
    super().__init__()
    assert isinstance(frameDim,int) and isinstance(chunkFrames,int)
    assert frameDim > 0 and chunkFrames > 0

    self.__chunkFrames = chunkFrames
    self.__frameBuffer = np.zeros([chunkFrames,frameDim],dtype="int16")
    self.__featurePIPE = PIPE()
    self.extract_function = None

    self.__extractThread = None
    self.__dim = None
    self.__reset_position_flag()
  
  def __reset_position_flag(self):
    self.__terminationStep = False
    self.__avaliableFrames = self.__chunkFrames

  @property
  def featDim(self):
    assert self.__dim is not None
    return self.__dim

  def __set_error(self):
    self.shift_state_to_error()
    self.__featurePIPE.set_error()
  
  def __set_termination(self):
    self.shift_state_to_termination()
    self.__featurePIPE.set_termination()

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
  
  def __compute_raw_feature(self,framePIPE):
    '''Compute raw feature'''
    print("Start extracting raw feature...")
    try:
      while True:
        # try to prepare chunk frames
        self.__frameBuffer.flags.writeable = True
        if not self.__prepare_chunk_frame(framePIPE):
          break
        self.__frameBuffer.flags.writeable = False
        # compute feature
        feats = self.extract_function(self.__frameBuffer[0:self.__avaliableFrames])
        # add to new PIPE
        if not self.__featurePIPE.is_alive():
          framePIPE.set_error()
          self.__set_error()
          break
        else: 
          for fid in range(self.__avaliableFrames-1):
              self.__featurePIPE.put( Vector(feats[fid],endpoint=False) )
          if self.__terminationStep:
            self.__featurePIPE.put( Vector(feats[self.__avaliableFrames-1], endpoint=True) )
            self.__reset_position_flag()
          else:
            self.__featurePIPE.put( Vector(feats[self.__avaliableFrames-1], endpoint=False) )
        # if no any data
        if framePIPE.is_exhaustion():
          self.__set_termination()
          break

    except Exception as e:
      framePIPE.set_error()
      self.__set_error()
      raise e
    finally:
      print("Stop extracting raw feature!")

  def start_extracting(self,framePIPE):

    assert self.extract_function is not None, "Please implement the feature extraction function."

    testData = np.ones_like(self.__frameBuffer)
    testResult = self.extract_function(testData)
    assert (isinstance(testResult,np.ndarray) and 
            len(testResult.shape) == 2 and 
            testResult.shape[0] == self.__chunkFrames), "The output of feature function must be a ( 1d -> 1 frame or 2d -> N frames) Numpy array."
    self.__dim = testResult.shape[-1]
    del testData
    del testResult

    self.__featureThread = threading.Thread(target=self.__compute_raw_feature,args=(framePIPE,))
    self.__featureThread.setDaemon(True)
    self.__featureThread.start()

  def get_feature_pipe(self):
    return self.__featurePIPE

class SpectrogramExtractor(FeatureExtractor):

  def __init__(self,frameDim=400,chunkFrames=10,
                energyFloor=0.0,rawEnergy=True,winType="povey",
                dither=0.0,removeDC=True,preemphCoeff=0.97,
                blackmanCoeff=0.42):
    super().__init__(frameDim,chunkFrames)
    assert isinstance(energyFloor,float) and energyFloor >= 0.0
    assert isinstance(dither,float) and dither >= 0.0
    assert isinstance(preemphCoeff,float) and 0 <= energyFloor <= 1
    assert isinstance(blackmanCoeff,float) and 0 < blackmanCoeff < 0.5
    assert isinstance(rawEnergy,bool)
    assert isinstance(removeDC,bool)

    self.__energy_floor = np.log(energyFloor) if energyFloor > 0 else 0
    self.__need_raw_energy = rawEnergy
    self.__remove_dc_offset = removeDC
    self.__preemph_coeff = preemphCoeff
    self.__dither_factor = dither
    self.__window = get_window_function(frameDim, winType, blackmanCoeff)
    self.extract_function = self.__extract_function

  def __extract_function(self,frames):
    
    if self.__dither_factor != 0: 
      frames = dither_singal_2d(frames, self.__dither_factor)
    if self.__remove_dc_offset: 
      frames = remove_dc_offset_2d(frames)
    if self.__need_raw_energy: 
      energies = compute_log_energy_2d(frames)
    if self.__preemph_coeff > 0:
      frames = pre_emphasize_2d(frames, self.__preemph_coeff)
    
    frames *= self.__window

    if not self.__need_raw_energy:
      energies = compute_log_energy_2d(frames)
    
    _, frames = split_radix_real_fft_2d(frames)
    frames = compute_power_spectrum_2d(frames)
    frames = apply_floor(frames)
    frames = np.log(frames)

    if self.__energy_floor != 0:
      energies[ energies < self.__energy_floor ] = self.__energy_floor

    frames[:,0] = energies

    return frames

class FbankExtractor(FeatureExtractor):

  def __init__(self,rate=16000,frameDim=400,chunkFrames=10,
                energyFloor=0.0,useEnergy=False,rawEnergy=True,winType="povey",
                dither=0.0,removeDC=True,preemphCoeff=0.97,
                blackmanCoeff=0.42,usePower=True,
                numBins=23,lowFreq=20,highFreq=0,useLog=True,
                ):
    super().__init__(frameDim,chunkFrames)
    assert isinstance(rate,int)
    assert isinstance(energyFloor,float) and energyFloor >= 0.0
    assert isinstance(dither,float) and dither >= 0.0
    assert isinstance(preemphCoeff,float) and 0 <= energyFloor <= 1
    assert isinstance(blackmanCoeff,float) and 0 < blackmanCoeff < 0.5
    assert isinstance(numBins,int) and numBins >= 3
    assert isinstance(lowFreq,int) and isinstance(highFreq,int) and lowFreq >= 0
    if highFreq != 0 :
      assert highFreq > lowFreq
    assert isinstance(useEnergy,bool)
    assert isinstance(rawEnergy,bool)
    assert isinstance(removeDC,bool)
    assert isinstance(usePower,bool)
    assert isinstance(useLog,bool)

    self.__energy_floor = np.log(energyFloor) if energyFloor > 0 else 0
    self.__add_energy = useEnergy
    self.__need_raw_energy = rawEnergy
    self.__remove_dc_offset = removeDC
    self.__preemph_coeff = preemphCoeff
    self.__dither = dither
    self.__usePower = usePower
    self.__useLog = useLog
    self.__window = get_window_function(frameDim,winType,blackmanCoeff)
    
    self.__fftLen = get_padded_fft_length(frameDim)
    self.__melInfo = (numBins,self.__fftLen,lowFreq,highFreq)
    self.__melFilters = get_mel_bins(numBins,rate,self.__fftLen,lowFreq,highFreq)

    self.extract_function = self.__extract_function

  def __extract_function(self,frames):
    
    if self.__dither != 0:
      frames = dither_singal_2d(frames, self.__dither)
    if self.__remove_dc_offset:
      frames = remove_dc_offset_2d(frames)
    if self.__add_energy and self.__need_raw_energy:
      energies = compute_log_energy_2d(frames)
    if self.__preemph_coeff:
      frames = pre_emphasize_2d(frames, self.__preemph_coeff)
    
    frames *= self.__window
    if self.__add_energy and not self.__need_raw_energy:
      energies = compute_log_energy_2d(frames)

    _, frames = split_radix_real_fft_2d(frames)
    frames = compute_power_spectrum_2d(frames)

    if not self.__usePower:
      frames = frames**0.5
    frames = np.dot( frames, self.__melFilters )
    
    if self.__useLog:
      frames = apply_floor(frames)
      frames = np.log(frames)

    if self.__add_energy:
      if self.__energy_floor != 0:
        energies[ energies < self.__energy_floor ] = self.__energy_floor
      frames = np.concatenate([energies[:,None],frames],axis=1)

    return frames

class MfccExtractor(FeatureExtractor):

  def __init__(self,rate=16000,frameDim=400,chunkFrames=10,
                energyFloor=0.0,useEnergy=True,rawEnergy=True,winType="povey",
                dither=0.0,removeDC=True,preemphCoeff=0.97,
                blackmanCoeff=0.42,
                numBins=23,lowFreq=20,highFreq=0,useLog=True,
                cepstralLifter=22,numCeps=13,
                ):
    super().__init__(frameDim,chunkFrames)
    assert isinstance(rate,int)
    assert isinstance(energyFloor,float) and energyFloor >= 0.0
    assert isinstance(dither,float) and dither >= 0.0
    assert isinstance(preemphCoeff,float) and 0 <= energyFloor <= 1
    assert isinstance(blackmanCoeff,float) and 0 < blackmanCoeff < 0.5
    assert isinstance(numBins,int) and numBins >= 3
    assert isinstance(lowFreq,int) and isinstance(highFreq,int) and lowFreq >= 0
    if highFreq != 0 :
      assert highFreq > lowFreq
    assert isinstance(cepstralLifter,int) and numBins >= 0
    assert isinstance(numCeps,int) and 0 < numCeps <= numBins
    assert isinstance(useEnergy,bool)
    assert isinstance(rawEnergy,bool)
    assert isinstance(removeDC,bool)
    assert isinstance(useLog,bool)    

    self.__energy_floor = np.log(energyFloor) if energyFloor > 0 else 0
    self.__use_energy = useEnergy
    self.__need_raw_energy = rawEnergy
    self.__remove_dc_offset = removeDC
    self.__preemph_coeff = preemphCoeff
    self.__dither = dither
    self.__useLog = useLog
    self.__window = get_window_function(frameDim,winType,blackmanCoeff)
    
    self.__fftLen = get_padded_fft_length(frameDim)
    self.__melInfo = (numBins,self.__fftLen,lowFreq,highFreq)
    self.__melFilters = get_mel_bins(numBins,rate,self.__fftLen,lowFreq,highFreq)

    self.__dctMat = get_dct_matrix(numCeps=numCeps,numBins=numBins)
    if cepstralLifter > 0:
      self.__cepsCoeff = get_cepstral_lifter_coeff(dim=numCeps,factor=cepstralLifter)
    else:
      self.__cepsCoeff = 1

    self.extract_function = self.__extract_function

  def __extract_function(self,frames):
    
    if self.__dither != 0:
      frames = dither_singal_2d(frames, self.__dither)
    if self.__remove_dc_offset:
      frames = remove_dc_offset_2d(frames)
    if self.__use_energy and self.__need_raw_energy:
      energies = compute_log_energy_2d(frames)
    if self.__preemph_coeff:
      frames = pre_emphasize_2d(frames, self.__preemph_coeff)
    
    frames *= self.__window
    if self.__use_energy and not self.__need_raw_energy:
      energies = compute_log_energy_2d(frames)

    _, frames = split_radix_real_fft_2d(frames)
    frames = compute_power_spectrum_2d(frames)

    frames = np.dot( frames, self.__melFilters )
    frames = apply_floor(frames)
    frames = np.log(frames)
    frames = frames.dot(self.__dctMat)
    frames = frames * self.__cepsCoeff

    if self.__use_energy:
      if self.__energy_floor != 0:
        energies[ energies < self.__energy_floor ] = self.__energy_floor
      frames[:,0] = energies

    return frames

class FeatureProcessor(StateFlag):

  def __init__(self,rawFeatDim,chunkFrames=32,
                    delta=0,deltaWindow=2,spliceLeft=0,spliceRight=0,
                    lda=None):
    super().__init__()
    assert isinstance(rawFeatDim,int) and rawFeatDim > 0
    assert isinstance(chunkFrames,int) and chunkFrames > 0
    assert isinstance(delta,int) and delta >= 0
    assert isinstance(deltaWindow,int) and deltaWindow > 0
    assert isinstance(spliceLeft,int) and spliceLeft >= 0
    assert isinstance(spliceRight,int) and spliceRight >= 0

    self.__dim = rawFeatDim
    self.__chunkFrames = chunkFrames
    self.__delta = delta
    self.__deltaWindow = deltaWindow
    self.__spliceLeft = spliceLeft
    self.__spliceRight = spliceRight

    if lda is not None:
      self.__ldaMat = load_lda_matrix(lda)
    else:
      self.__ldaMat = None

    self.__center = chunkFrames
    self.__left = delta + spliceLeft
    self.__right = delta + spliceRight
    self.__cover = self.__left + self.__right
    self.__width = self.__center + self.__left + self.__right

    self.__featureBuffer = np.zeros([self.__width,rawFeatDim],dtype="float32")
    self.__newFeaturePIPE = PIPE()
    self.process_function = None
    self.__processThread = None

    self.__dim = None
    self.__reset_position_flag()

  def __reset_position_flag(self):
    self.__zerothStep = True
    self.__firstStep = False
    self.__terminationStep = False
    self.__avaliableFrames = self.__width

  @property
  def featDim(self):
    assert self.__dim is not None
    return self.__dim

  def __default_process_function(self, feats):

    #if self.__cmvn: 
    #    feat = self.use_cmvn(feat)

    if self.__delta > 0: 
      feats = add_deltas(feats,order=self.__delta,window=self.__deltaWindow)
    if self.__spliceLeft != 0 or self.__spliceRight != 0: 
      feats = splice_feats(feats,left=self.__spliceLeft,right=self.__spliceRight)
    if self.__ldaMat: 
      feats = feats.dot(self.__ldaMat)

    return feats
  
  def get_feature_pipe(self):
    return self.__newFeaturePIPE

  def __set_error(self):
    self.shift_state_to_error()
    self.__newFeaturePIPE.set_error()

  def __set_termination(self):
    self.shift_state_to_termination()
    self.__newFeaturePIPE.set_termination()

  def __prepare_chunk_feature(self,rawFeaturePIPE):
    '''
    Prepare chunk stream to compute feature.
    '''
    timeCost = 0
    addedNewData = False 

    # move history data
    if self.__zerothStep:
      pos = 0
      self.__zerothStep = False
      self.__firstStep = True
    else:
      self.__featureBuffer[0:self.__cover,:] = self.__featureBuffer[ -self.__cover:,: ]
      pos = self.__cover
      self.__firstStep = False

    while pos < self.__width:
      if rawFeaturePIPE.is_error():
        self.__set_error()
        return False
      elif rawFeaturePIPE.is_exhaustion():
        self.__avaliableFrames = pos
        self.__terminationStep = True
        break
      elif rawFeaturePIPE.is_empty():
        time.sleep(TIMESCALE)
        timeCost += TIMESCALE
        if timeCost > TIMEOUT:
          rawFeaturePIPE.set_error()
          self.__set_error()
          return False
      else:
        vec = rawFeaturePIPE.get()
        assert isinstance(vec,Vector)
        if addedNewData:
          self.__featureBuffer[pos,:] = vec.item
          pos += 1
          if vec.is_endpoint():
            self.__avaliableFrames = pos
            self.__terminationStep = True
            break
        else:
          if vec.is_endpoint():
            continue # discard this element
          self.__featureBuffer[pos,:] = vec.item
          pos += 1
          addedNewData = True
    
    if pos == 0:
      self.__set_termination()
      return False
    else:
      self.__featureBuffer[pos:,:] = 0
    
    return True

  def __process_feature(self,rawFeaturePIPE):
    
    print("Start processing raw feature...")
    try:
      while True:
        # try to prepare chunk frames
        self.__featureBuffer.flags.writeable = True
        if not self.__prepare_chunk_feature(rawFeaturePIPE): 
          break
        self.__featureBuffer.flags.writeable = False
        # process
        feats = self.process_function(self.__featureBuffer[:self.__avaliableFrames,:])
        self.__dim = feats.shape[-1]
        # add to PIPE
        if not self.__newFeaturePIPE.is_alive():
          rawFeaturePIPE.set_error()
          self.__set_error()
          break

        if self.__firstStep:
          realFeats = feats[0:-self.__right,:]
        elif self.__terminationStep:
          realFeats = feats[self.__left:,:]
        else:
          realFeats = feats[self.__left:-self.__right,:]

        realLength = len(realFeats)
        
        for fid in range(realLength-1):
          self.__newFeaturePIPE.put( Vector(realFeats[fid],endpoint=False) )

        if self.__terminationStep:
          self.__newFeaturePIPE.put( Vector(realFeats[realLength-1],endpoint=True) )
          self.__reset_position_flag()
        else:
          self.__newFeaturePIPE.put( Vector(realFeats[realLength-1],endpoint=False) )

    except Exception as e:
      rawFeaturePIPE.set_error()
      self.__set_error()
      raise e
    finally:
      print("Stop processing raw feature!")

  def start_processing(self,rawFeaturePIPE):

    if self.process_function is None:
      self.process_function = self.__default_process_function
    
    self.__processThread = threading.Thread(target=self.__process_feature, args=(rawFeaturePIPE,))
    self.__processThread.setDaemon(True)
    self.__processThread.start()