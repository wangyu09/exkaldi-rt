# coding=utf-8
#
# Yu Wang (University of Yamanashi)
# Apr, 2021
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
import webrtcvad
import threading
import multiprocessing
import numpy as np
import subprocess
from collections import namedtuple

#from exkaldirt.base import ExKaldiRTBase, Component, PIPE, Tunnel, Packet
#from exkaldirt.utils import run_exkaldi_shell_command
#from exkaldirt.base import info
#from exkaldirt.base import ENDPOINT, is_endpoint, NullPIPE

from base import ExKaldiRTBase, Component, PIPE, Tunnel, Packet, ContextManager
from utils import run_exkaldi_shell_command, encode_vector_temp
from base import info, mark, print_
from base import ENDPOINT, is_endpoint, NullPIPE

import sys
sys.path.append( info.CMDROOT )
import cutils

###############################################
# 1. Some functions for feature extraction
###############################################

def pre_emphasize_1d(waveform,coeff=0.95):
  '''
  Pre-emphasize the waveform.

  Args:
    _waveform_: (1-d np.ndarray) The waveform data.
    _coeff_: (float) Coefficient. 0 <= coeff < 1.0. 
  
  Return:
    A new 1-d np.ndarray.
  '''
  assert 0 <= coeff < 1.0
  assert isinstance(waveform,np.ndarray) and  len(waveform.shape) == 1
  new = np.zeros_like(waveform)
  new[1:] = waveform[1:] - coeff*waveform[:-1]
  new[0] = waveform[0] - coeff*waveform[0]
  return new

def pre_emphasize_2d(waveform,coeff=0.95):
  '''
  Pre-emphasize the waveform.

  Args:
    _waveform_: (2-d np.ndarray) A batch of waveform data.
    _coeff_: (float) Coefficient. 0 <= coeff < 1.0. 
  
  Return:
    A new 2-d np.ndarray.
  '''
  assert 0 <= coeff < 1.0
  assert isinstance(waveform,np.ndarray) and  len(waveform.shape) == 2
  new = np.zeros_like(waveform)
  new[:,1:] = waveform[:,1:] - coeff*waveform[:,:-1]
  new[:,0] = waveform[:,0] - coeff*waveform[:,0]
  return new

def get_window_function(size,winType="povey",blackmanCoeff=0.42):
  '''
  Get a window.

  Args:
    _size_: (int) The width of window.
    _winType_: (str) Window type. "hanning", "sine", "hamming", "povey", "rectangular" or "blackman".
  
  Return:
    A 1-d np.ndarray.
  '''
  assert isinstance(size,int) and size > 0
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
      assert isinstance(blackmanCoeff,float)
      winType[i] = blackmanCoeff - 0.5*np.cos(a*i) + (0.5-blackmanCoeff)*np.cos(2*a*i)
    else:
      raise Exception(f"Unknown Window Type: {winType}")
  
  return window

def dither_singal_1d(waveform,factor=1.0):
  '''
  Dither the signal.

  Args:
    _waveform_: (1-d np.ndarray) The waveform.
    _factor_: (float) Dither factor.
  
  Return:
    A new 1-d np.ndarray.
  '''
  assert isinstance(waveform,np.ndarray) and len(waveform.shape) == 1
  return cutils.dither( waveform[None,:], factor)[0]

def dither_singal_2d(waveform,factor=0.0):
  '''
  Dither the signal.

  Args:
    _waveform_: (2-d np.ndarray) A batch of waveforms.
    _factor_: (float) Dither factor.
  
  Return:
    A new 2-d np.ndarray.
  '''
  assert isinstance(waveform,np.ndarray) and len(waveform.shape) == 2
  return cutils.dither(waveform, factor)

def remove_dc_offset_1d(waveform):
  '''
  Remove the direct current offset.

  Args:
    _waveform_: (1-d np.ndarray) The waveform.
  
  Return:
    A new 1-d np.ndarray.
  '''
  assert isinstance(waveform,np.ndarray) and len(waveform.shape) == 1
  return waveform - np.mean(waveform)

def remove_dc_offset_2d(waveform):
  '''
  Remove the direct current offset.

  Args:
    _waveform_: (2-d np.ndarray) A batch of waveforms.
  
  Return:
    A new 2-d np.ndarray.
  '''
  assert isinstance(waveform,np.ndarray) and len(waveform.shape) == 2
  return waveform - np.mean(waveform,axis=1,keepdims=True)

def compute_log_energy_1d(waveform,floor=info.EPSILON):
  '''
  Compute log energy.

  Args:
    _waveform_: (1-d np.ndarray) The waveform.
    _floor_: (float) Float floor value. 
  
  Return:
    A new 1-d np.ndarray.
  '''
  assert isinstance(waveform,np.ndarray) and len(waveform.shape) == 1
  return np.log(max(np.sum(waveform**2),floor))

def compute_log_energy_2d(waveform,floor=info.EPSILON):
  '''
  Compute log energy.

  Args:
    _waveform_: (2-d np.ndarray) A batch of waveforms.
    _floor_: (float) Float floor value. 
  
  Return:
    A new 2-d np.ndarray.
  '''
  assert isinstance(waveform,np.ndarray) and len(waveform.shape) == 2
  temp = np.sum(waveform**2,axis=1)
  temp[temp < floor] = floor
  return np.log(temp)

def split_radix_real_fft_1d(waveform):
  '''
  Compute split radix FFT.

  Args:
    _waveform_: (1-d np.ndarray) The waveform.
  
  Return:
    A tuple: (FFT length, Result ).
    _FFT length_: (int).
    _Result_: (A 2-d np.ndarray) The first dimension is real values, The second dimension is image values.
  '''
  assert isinstance(waveform,np.ndarray) and len(waveform.shape) == 1
  points = len(waveform)
  fftLen = get_padded_fft_length(points)
  result = cutils.srfft(waveform[None,:],fftLen)[0]
  return fftLen, result

def split_radix_real_fft_2d(waveform):
  '''
  Compute split radix FFT.

  Args:
    _waveform_: (2-d np.ndarray) A batch of waveforms.
  
  Return:
    A tuple: ( FFT length, Result ).
    _FFT length_: (int).
    _Result_: (A 3-d np.ndarray) The 2st dimension is real values, The 3st dimension is image values.
  '''
  assert isinstance(waveform,np.ndarray) and len(waveform.shape) == 2
  points = waveform.shape[1]
  fftLen = get_padded_fft_length(points)
  result = cutils.srfft(waveform,fftLen)
  return fftLen, result

def compute_power_spectrum_1d(fftFrame):
  '''
  Compute power spectrum.

  Args:
    _fftFrame_: (2-d np.ndarray) A frame of FFT result.
  
  Return:
    A 1-d np.ndarray.
  '''
  assert isinstance(fftFrame,np.ndarray) and len(fftFrame.shape) == 2
  
  zeroth = fftFrame[0,0] + fftFrame[0,1]
  n2th = fftFrame[0,0] - fftFrame[0,1]
  
  fftFrame = np.sum(fftFrame**2,axis=1)
  fftFrame[0] = zeroth**2

  return np.append(fftFrame,n2th**2)

def compute_power_spectrum_2d(fftFrame):
  '''
  Compute power spectrum.

  Args:
    _fftFrame_: (2-d np.ndarray) A batch of frames. FFT results.
  
  Return:
    A 2-d np.ndarray.
  '''
  assert isinstance(fftFrame,np.ndarray) and len(fftFrame.shape) == 3
  
  zeroth = fftFrame[:,0,0] + fftFrame[:,0,1]
  n2th = fftFrame[:,0,0] - fftFrame[:,0,1]
  
  fftFrame = np.sum(fftFrame**2,axis=2)
  fftFrame[:,0] = zeroth**2

  return np.append(fftFrame,(n2th**2)[:,None],axis=1)

def apply_floor(feature,floor=info.EPSILON):
  '''
  Apply float floor to feature.

  Args:
    _feature_: (np.ndarray) Feature.
    _floor_: (float) Float floor value.
  
  Return:
    A 2-d np.ndarray (Not new).
  '''
  feature[feature<floor] = floor
  return feature

def mel_scale(freq):
  '''
  Do Mel scale.

  Args:
    _freq_: (int) Frequency.
  
  Return:
    A float value.
  '''
  return 1127.0 * np.log (1.0 + freq / 700.0)

def inverse_mel_scale(melFreq):
  '''
  Do Inverse Mel scale.

  Args:
    _freq_: (int) Frequency.
  
  Return:
    A float value.
  '''
  return 700.0 * (np.exp(melFreq/1127.0) - 1)

def get_mel_bins(numBins,rate,fftLen,lowFreq=20,highFreq=0):
  '''
  Get the Mel filters bank.

  Args:
    _numBins_: (int) The number of filters.
    _rate_: (int) Sampling rate.
    _fftLen_: (int) FFT length.
    _lowFreq_: (int) The minimum frequency.
    _highFreq_: (int) The maximum frequency. If zero, highFreq = rate/2. If < 0, highFreq = rate/2 - highFreq.
  
  Return:
    A 2-d np.ndarray with shape ( fftLen/2, numBins ).
  '''
  assert isinstance(numBins,int) and numBins >= 0
  assert isinstance(rate,int) and rate >= 2
  assert isinstance(fftLen,int) and fftLen > 0 and int(np.log2(fftLen)) == np.log2(fftLen)
  assert isinstance(lowFreq,int) and lowFreq >= 0
  assert isinstance(highFreq,int)

  nyquist = int(0.5 * rate)
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
  '''
  Compute FFT length.

  Args:
    _points_: (int) Frame width.
  
  Return:
    An int value.
  '''
  assert isinstance(points,int) and points >= 2
  fftLen = 1
  while fftLen < points:
    fftLen <<= 1
  return fftLen

def get_dct_matrix(numCeps,numBins):
  '''
  Compute DCT matrix.

  Args:
    _numCeps_: (int) The dim. of MFCC.
    _numBins_: (int) The dim. of fBank.
  
  Return:
    An 2-d np.ndarray with shape: (numBins, numCeps)
  '''
  assert isinstance(numCeps,int) and numCeps > 0
  assert isinstance(numBins,int) and numBins > 0

  result = np.zeros([numCeps,numBins],dtype="float32")
  result[0] = np.sqrt(1/numBins)
  normalizer = np.sqrt(2/numBins)
  for i in range(1,numCeps):
    for j in range(0,numBins):
      result[i,j] = normalizer * np.cos( np.pi/numBins*(j+0.5)*i )
  return result.T

def get_cepstral_lifter_coeff(dim,factor=22):
  '''
  Compute cepstral lifter coefficient.

  Args:
    _dim_: (int) The dim. of MFCC.
    _factor_: (int) Factor.
  
  Return:
    A 1-d np.ndarray.
  '''
  assert isinstance(dim,int) and dim > 0
  assert factor > 0
  result = np.zeros([dim,],dtype="float32")
  for i in range(dim):
    result[i] = 1.0 + 0.5*factor*np.sin(np.pi*i/factor)
  return result

def add_deltas(feat, order=2, window=2):
  '''
  Append delta feature.

  Args:
    _feat_: (2-d np.ndarray) Feature with shape (frames, dim).
    _order_: (int).
    _window_: (int).
  
  Return:
    An new 2-d np.ndarray with shape: (frames, dim * (order+1)).
  '''
  assert isinstance(feat,np.ndarray) and len(feat.shape) == 2
  assert isinstance(order,int) and order > 0
  assert isinstance(window,int) and window > 0

  # return cutils.add_deltas(feat,order,window)

  frames = feat.shape[0]
  dims = feat.shape[1]
  inputs = f"{frames} {dims} ".encode() + encode_vector_temp( feat.reshape(-1) )
  cmd = os.path.join(info.CMDROOT,f"exkaldi-add-deltas --order {order} --window {window}")
  out = run_exkaldi_shell_command(cmd,inputs=inputs)
  return np.array(out,dtype="float32").reshape([frames,-1])

def splice_feats(feat, left, right):
  '''
  Splice the left and right context of feature.

  Args:
    _feat_: (2-d np.ndarray) Feature with shape (frames, dim).
    _left_: (int).
    _right_: (int).
  
  Return:
    An new 2-d np.ndarray with shape: (frames, dim * (left+right+1)).
  '''
  assert isinstance(feat,np.ndarray) and len(feat.shape) == 2
  assert isinstance(left,int) and left >= 0
  assert isinstance(right,int) and right >= 0
  if left == 0 and right ==0: return feat
  return cutils.splice_feat(feat, left, right)

# This function is wrapped from kaldi_io library.
def load_lda_matrix(ldaFile):
  '''
  Read the LDA(+MLLT) matrix from Kaldi file.

  Args:
    _ldaFile_: (str) LDA matrix file path.
  '''
  assert os.path.isfile(ldaFile), f"No such file: {ldaFile}."
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

class MatrixFeatureExtractor(Component):
  '''
  The base class of a feature extractor.
  Please implement the self.extract_function by your function.
  '''
  def __init__(self,extFunc,minParallelSize=10,oKey="data",name=None):
    '''
    Args:
      _frameDim_: (int) The dim. of frame.
      _batchSize_: (int) Batch size.
      _minParallelSize_: (int) If _batchSize_ >= minParallelSize, run two parallel threads for one batch features.
      _name_: (str) Name.
    '''
    super().__init__(oKey=oKey,name=name)
    assert isinstance(minParallelSize,int) and minParallelSize >= 2
    assert callable(extFunc)
    self.__extract_function_ = extFunc
    self.__minParallelBatchSize = minParallelSize//2

  def core_loop(self):

    self.__firstStep = True

    while True:

      action = self.decide_action()
      self.__featureCache = [None,None]

      if action is False:
        break
      elif action is None:
        self.outPIPE.stop()
        break
      else:
        packet = self.get_packet()
        if is_endpoint(packet):
          self.put_packet( ENDPOINT )
          continue
        else:
          iKey = packet.mainKey if self.iKey is None else self.iKey
          mat = packet[ iKey ]
          assert isinstance(mat, np.ndarray) and len(mat.shape) == 2

          bsize = len(mat)
          if len(mat) < self.__minParallelBatchSize:
            newMat = self.__extract_function_( mat )
          else:
            mid = bsize // 2
            ### open thread 1 to compute first half part
            thread1 = threading.Thread(target=self.__extract_parallel,args=(mat[0:mid],0,))
            thread1.setDaemon(True)
            thread1.start()
            ### open thread 2 to compute second half part
            thread2 = threading.Thread(target=self.__extract_parallel,args=(mat[mid:],1,))
            thread2.setDaemon(True)
            thread2.start()
          
            thread1.join()
            thread2.join()
            ### Concat
            newMat = np.concatenate(self.__featureCache,axis=0)

          if self.__firstStep:
            assert (isinstance(newMat,np.ndarray) and len(newMat.shape) == 2) ,\
                  "The output of feature function must be a ( 1d -> 1 frame or 2d -> N frames) Numpy array."
            if newMat.shape[0] != bsize:
              print(f"{self.name}: Warning! The frames of features is lost.")

          ## Append feature into PIPE if necessary.
          packet.add(key=self.oKey[0],data=newMat,asMainKey=True)
          self.put_packet( packet )

  def __extract_parallel(self,featChunk,ID):
    '''
    A thread function to compute feature.
    '''
    self.__featureCache[ID] = self.__extract_function_(featChunk)

class SpectrogramExtractor(MatrixFeatureExtractor):
  '''
  Spectrogram feature extractor. 
  '''
  def __init__(self,energyFloor=0.0,rawEnergy=True,winType="povey",
                dither=1.0,removeDC=True,preemphCoeff=0.97,
                blackmanCoeff=0.42,minParallelSize=10,
                oKey="data",name=None):
    '''
    Args:
      _frameDim_: (int) The dim. of frame.
      _batchSize_: (int) Batch size.
      _energyFloor_: (float) The energy floor value.
      _rawEnergy_: (bool) If True, compute energy from raw waveform.
      _winType_: (str) Window type. "hanning", "sine", "hamming", "povey", "rectangular" or "blackman".
      _dither_: (float) Dither factor.
      _removeDC_: (bool) If True remove DC offset.
      _preemphCoeff_: (float) Pre-emphasize factor.
      _blackmanCoeff_: (float) Blackman window coefficient.
      _minParallelSize_: (int) If _batchSize_ >= minParallelSize, run two threads to extract feature.
      _name_: (str) None.
    '''
    super().__init__(extFunc=self.__extract_function,minParallelSize=minParallelSize,oKey=oKey,name=name)

    assert isinstance(energyFloor,float) and energyFloor >= 0.0
    assert isinstance(rawEnergy,bool)
    assert isinstance(dither,float) and dither >= 0.0
    assert isinstance(removeDC,bool)
    assert isinstance(preemphCoeff,float) and 0 <= energyFloor <= 1
    assert isinstance(blackmanCoeff,float) and 0 < blackmanCoeff < 0.5
    self.__energy_floor = np.log(energyFloor) if energyFloor > 0 else 0
    self.__need_raw_energy = rawEnergy
    self.__remove_dc_offset = removeDC
    self.__preemph_coeff = preemphCoeff
    self.__dither_factor = dither

    self.__winType = winType
    self.__blackmanCoeff = blackmanCoeff
    self.__window = None

  def __extract_function(self,frames):

    if self.__window is None:
      frameDim = frames.shape[0]
      self.__window = get_window_function(frameDim, self.__winType, self.__blackmanCoeff)
    
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

class FbankExtractor(MatrixFeatureExtractor):
  '''
  FBank feature extractor. 
  '''
  def __init__(self,rate=16000,
                energyFloor=0.0,useEnergy=False,rawEnergy=True,winType="povey",
                dither=1.0,removeDC=True,preemphCoeff=0.97,
                blackmanCoeff=0.42,usePower=True,
                numBins=23,lowFreq=20,highFreq=0,useLog=True,
                minParallelSize=10,
                oKey="data",name=None):
    '''
    Args:
      _rate_: (int) Sampling rate.
      _frameDim_: (int) The dim. of frame.
      _batchSize_: (int) Batch size.
      _energyFloor_: (float) The energy floor value.
      _useEnergy_: (bool) If True, Add energy dim. to the final fBank feature.
      _rawEnergy_: (bool) If True, compute energy from raw waveform.
      _winType_: (str) Window type. "hanning", "sine", "hamming", "povey", "rectangular" or "blackman".
      _dither_: (float) Dither factor.
      _removeDC_: (bool) If True remove DC offset.
      _preemphCoeff_: (float) Pre-emphasize factor.
      _blackmanCoeff_: (float) Blackman window coefficient.
      _usePower_: (bool) If True, use power spectrogram.
      _numBins_: (int) The dim. of fBank feature.
      _lowFreq_: (int) The minimum frequency.
      _lowFreq_: (int) The maximum frequency.
      _useLog_: (bool) If True, compute log fBank.
      _minParallelSize_: (int) If _batchSize_ >= minParallelSize, run two threads to extract feature.
      _name_: (str) None.
    '''        
    super().__init__(extFunc=self.__extract_function,minParallelSize=minParallelSize,oKey=oKey,name=name)
    assert isinstance(rate,int) and rate > 0
    assert isinstance(energyFloor,float) and energyFloor >= 0.0
    assert isinstance(useEnergy,bool)
    assert isinstance(rawEnergy,bool)
    assert isinstance(dither,float) and dither >= 0.0
    assert isinstance(removeDC,bool)
    assert isinstance(preemphCoeff,float) and 0 <= energyFloor <= 1
    assert isinstance(blackmanCoeff,float) and 0 < blackmanCoeff < 0.5
    assert isinstance(numBins,int) and numBins >= 3
    assert isinstance(lowFreq,int) and isinstance(highFreq,int) and lowFreq >= 0
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
  
    self.__winInfo = (winType, blackmanCoeff)
    self.__window = None
    self.__melInfo = (numBins,rate,lowFreq,highFreq)
    self.__melFilters = None

  def __extract_function(self,frames):
    
    if self.__window is None:
      frameDim = frames.shape[1]
      self.__window = get_window_function(frameDim, 
                                          self.__winInfo[0], 
                                          self.__winInfo[1],
                                        )
      fftLen = get_padded_fft_length(frameDim)
      self.__melFilters = get_mel_bins(self.__melInfo[0],
                                       self.__melInfo[1],
                                       fftLen,
                                       self.__melInfo[2],
                                       self.__melInfo[3],
                                      )

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

class MfccExtractor(MatrixFeatureExtractor):
  '''
  MFCC feature extractor. 
  '''
  def __init__(self,rate=16000,
                energyFloor=0.0,useEnergy=True,rawEnergy=True,winType="povey",
                dither=1.0,removeDC=True,preemphCoeff=0.97,
                blackmanCoeff=0.42,
                numBins=23,lowFreq=20,highFreq=0,useLog=True,
                cepstralLifter=22,numCeps=13,
                minParallelSize=10,
                oKey="data",name=None):
    '''
    Args:
      _rate_: (int) Sampling rate.
      _frameDim_: (int) The dim. of frame.
      _batchSize_: (int) Batch size.
      _energyFloor_: (float) The energy floor value.
      _useEnergy_: (bool) If True, Replace the first dim. of feature with energy.
      _rawEnergy_: (bool) If True, compute energy from raw waveform.
      _winType_: (str) Window type. "hanning", "sine", "hamming", "povey", "rectangular" or "blackman".
      _dither_: (float) Dither factor.
      _removeDC_: (bool) If True remove DC offset.
      _preemphCoeff_: (float) Pre-emphasize factor.
      _blackmanCoeff_: (float) Blackman window coefficient.
      _numBins_: (int) The dim. of fBank feature.
      _lowFreq_: (int) The minimum frequency.
      _lowFreq_: (int) The maximum frequency.
      _useLog_: (bool) If True, compute log fBank.
      _cepstralLifter_: (int) MFCC lifter factor.
      _numCeps_: (int) The dim. of MFCC feature.
      _minParallelSize_: (int) If _batchSize_ >= minParallelSize, run two threads to extract feature.
      _name_: (str) None.
    '''     
    super().__init__(extFunc=self.__extract_function,minParallelSize=minParallelSize,oKey=oKey,name=name)
    assert isinstance(rate,int)
    assert isinstance(energyFloor,float) and energyFloor >= 0.0
    assert isinstance(dither,float) and dither >= 0.0
    assert isinstance(preemphCoeff,float) and 0 <= energyFloor <= 1
    assert isinstance(blackmanCoeff,float) and 0 < blackmanCoeff < 0.5
    assert isinstance(numBins,int) and numBins >= 3
    assert isinstance(lowFreq,int) and isinstance(highFreq,int) and lowFreq >= 0
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
    
    self.__winInfo = (winType, blackmanCoeff)
    self.__window = None
    self.__melInfo = (numBins,rate,lowFreq,highFreq)
    self.__melFilters = None

    self.__dctMat = get_dct_matrix(numCeps=numCeps,numBins=numBins)
    if cepstralLifter > 0:
      self.__cepsCoeff = get_cepstral_lifter_coeff(dim=numCeps,factor=cepstralLifter)
    else:
      self.__cepsCoeff = 1

  def __extract_function(self,frames):
  
    if self.__window is None:
      frameDim = frames.shape[1]
      self.__window = get_window_function(frameDim, 
                                          self.__winInfo[0], 
                                          self.__winInfo[1],
                                        )
      fftLen = get_padded_fft_length(frameDim)
      self.__melFilters = get_mel_bins(self.__melInfo[0],
                                       self.__melInfo[1],
                                       fftLen,
                                       self.__melInfo[2],
                                       self.__melInfo[3],
                                      )

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

class MixtureExtractor(MatrixFeatureExtractor):
  '''
  Mixture feature extractor.
  You can extract Mixture of "spectrogram", "fbank" and "mfcc" in the same time. 
  '''
  def __init__(self,
                mixType=["mfcc","fbank"],
                rate=16000,dither=0.0,rawEnergy=True,winType="povey",
                removeDC=True,preemphCoeff=0.97,
                blackmanCoeff=0.42,energyFloor=0.0,
                numBins=23,lowFreq=20,highFreq=0,
                useEnergyForFbank=True,
                usePowerForFbank=True,
                useLogForFbank=True,
                useEnergyForMfcc=True,
                cepstralLifter=22,numCeps=13,
                minParallelSize=10,oKeys=None,name=None):

    # Check the mixture type
    assert isinstance(mixType,(list,tuple)), f"{self.name}: <mixType> should be a list or tuple."
    for featType in mixType:
      assert featType in ["mfcc","fbank","spectrogram"], f'{self.name}: <mixType> should be "mfcc","fbank","spectrogram".' 
    assert len(mixType) == len(set(mixType)) and len(mixType) > 1
    self.__mixType = mixType

    if oKeys is None:
      oKeys = mixType
    else:
      assert isinstance(oKeys,(tuple,list)) and len(oKeys) == len(mixType)

    super().__init__(extFunc=self.__extract_function,
                     minParallelSize=minParallelSize,oKey=oKeys,name=name)

    # Some parameters for basic computing
    assert isinstance(rate,int)
    assert isinstance(dither,float) and dither >= 0.0
    self.__dither_factor = dither
    assert isinstance(removeDC,bool)
    self.__remove_dc_offset = removeDC
    assert isinstance(rawEnergy,bool)
    self.__need_raw_energy = rawEnergy
    assert isinstance(preemphCoeff,float) and 0 <= energyFloor <= 1
    self.__preemph_coeff = preemphCoeff
    assert isinstance(blackmanCoeff,float) and 0 < blackmanCoeff < 0.5
    self.__winInfo = (winType,blackmanCoeff)
    self.__window = None
    assert isinstance(energyFloor,float) and energyFloor >= 0.0
    self.__energy_floor = np.log(energyFloor) if energyFloor > 0 else 0 #????
    
    # Some parameters for fbank
    assert isinstance(numBins,int) and numBins >= 3
    assert isinstance(lowFreq,int) and isinstance(highFreq,int) and lowFreq >= 0
    if highFreq != 0 :
      assert highFreq > lowFreq
    self.__fftLen = None
    self.__melInfo = (rate,numBins,lowFreq,highFreq)
    self.__melFilters = None
    assert isinstance(useEnergyForFbank,bool)
    self.__use_energy_fbank = useEnergyForFbank
    assert isinstance(useLogForFbank,bool)
    self.__use_log_fbank = useLogForFbank
    assert isinstance(usePowerForFbank,bool)
    self.__use_power_fbank = usePowerForFbank

    # Some parameters for mfcc
    assert isinstance(cepstralLifter,int) and numBins >= 0
    assert isinstance(numCeps,int) and 0 < numCeps <= numBins
    assert isinstance(useEnergyForMfcc,bool)
    self.__use_energy_mfcc = useEnergyForMfcc
    self.__dctMat = get_dct_matrix(numCeps=numCeps,numBins=numBins)
    if cepstralLifter > 0:
      self.__cepsCoeff = get_cepstral_lifter_coeff(dim=numCeps,factor=cepstralLifter)
    else:
      self.__cepsCoeff = 1

  def __extract_function(self,frames):

    if self.__window is None:
      frameDim = frames.shape[1]
      self.__window = get_window_function(frameDim, 
                                          self.__winInfo[0], 
                                          self.__winInfo[1],
                                        )
      fftLen = get_padded_fft_length(frameDim)
      self.__melFilters = get_mel_bins(self.__melInfo[0],
                                       self.__melInfo[1],
                                       fftLen,
                                       self.__melInfo[2],
                                       self.__melInfo[3],
                                      )

    # Dither singal
    if self.__dither_factor != 0: 
      frames = dither_singal_2d(frames, self.__dither_factor)
    # Remove dc offset
    if self.__remove_dc_offset: 
      frames = remove_dc_offset_2d(frames)
    # Compute raw energy
    if self.__need_raw_energy: 
      energies = compute_log_energy_2d(frames)
    # Pre-emphasize
    if self.__preemph_coeff > 0:
      frames = pre_emphasize_2d(frames, self.__preemph_coeff)
    # Add window
    frames *= self.__window
    # Compute energy
    if not self.__need_raw_energy:
      energies = compute_log_energy_2d(frames)
    # Apply energy floor
    if self.__energy_floor != 0:
      energies[ energies < self.__energy_floor ] = self.__energy_floor
    # FFT
    _, frames = split_radix_real_fft_2d(frames)
    # Power spectrogram
    frames = compute_power_spectrum_2d(frames)
    
    outFeats = dict((name,None) for name in self.__mixType)
    # Compute the spectrogram feature
    if "spectrogram" in self.__mixType:
      specFrames = frames.copy()
      specFrames = apply_floor( specFrames )
      specFrames = np.log( specFrames )
      specFrames[:,0] = energies
      outFeats["spectrogram"] = specFrames

    # Compute the fbank feature
    if "fbank" in self.__mixType:
      fbankFrames = frames.copy()
      if not self.__use_power_fbank:
        fbankFrames = fbankFrames**0.5
      fbankFrames = np.dot( fbankFrames, self.__melFilters )
      if self.__use_log_fbank:
        fbankFrames = apply_floor(fbankFrames)
        fbankFrames = np.log(fbankFrames)
      if self.__use_energy_fbank:
        fbankFrames = np.concatenate([energies[:,None],fbankFrames],axis=1)
      outFeats["fbank"] = fbankFrames

    # Compute the mfcc feature
    if "mfcc" in self.__mixType:
      mfccFeats = frames
      mfccFeats = np.dot( mfccFeats, self.__melFilters )
      mfccFeats = apply_floor( mfccFeats )
      mfccFeats = np.log( mfccFeats )
      mfccFeats = mfccFeats.dot( self.__dctMat )
      mfccFeats = mfccFeats * self.__cepsCoeff
      if self.__use_energy_mfcc:
        mfccFeats[:,0] = energies
      outFeats["mfcc"] = mfccFeats

    # Merge the features
    finalFeats = []
    for featType in self.__mixType:
      finalFeats.append( outFeats[featType] )
  
    return np.concatenate(finalFeats, axis=1)

###############################################
# 2. Some functions for Online CMVN
###############################################

def compute_spk_stats(feats):
  '''
  Compute the statistics from speaker utterances.
  
  Args:
    _feats_: (2-d array, list or tuple) All utterances of a speaker.
  
  Return:
    A 2-d array with shape (2, feat dim + 1)
  '''
  if not isinstance(feats,(list,tuple)):
    feats = [feats,]
  dim = None
  stats = None
  for feat in feats:
    assert isinstance(feat,np.ndarray) and len(feat.shape) == 2, "<feats> should be 2-d NumPy array." 
    if dim is None:
      dim = feat.shape[1]
      stats = np.zeros([2,dim+1],dtype=feat.dtype)
    else:
      assert dim == feat.shape[1], "Feature dims do not match!"
    stats[0,0:dim] += np.sum(feat,axis=0)
    stats[1,0:dim] += np.sum(feat**2,axis=0)
    stats[0,dim] += len(feat)
  
  return stats
    
def get_kaldi_cmvn(fileName,spk=None):
  '''
  get the global(or speaker) CMVN from Kaldi cmvn statistics file.
  
  Args:
    _fileName_: (str) Kaldi cmvn .ark file.
    _spk_: (str) Speaker ID.
  
  Return:
    A 2-d array.
  '''
  assert os.path.isfile(fileName), f"No such file: {fileName} ."
  assert spk is None or isinstance(spk,str), f"<spk> should be a string."

  result = None
  with open(fileName, 'rb') as fp:
    while True:
      # read utterance ID
      utt = ''
      while True:
        char = fp.read(1).decode()
        if (char == '') or (char == ' '):break
        utt += char
      utt = utt.strip()
      if utt == '':
        if fp.read() == b'': break
        else: raise Exception("Miss utterance ID before utterance in stats file.")
      # read binary symbol
      binarySymbol = fp.read(2).decode()
      if binarySymbol == '\0B':
        sizeSymbol = fp.read(1).decode()
        if sizeSymbol not in ["C","F","D"]:
          raise Exception(f"Missed format flag. This might not be a kaldi stats file.")
        dataType = sizeSymbol + fp.read(2).decode() 
        if dataType == 'CM ':
          raise Exception("Unsupported to read compressed binary kaldi matrix data.")                    
        elif dataType == 'FM ':
          sampleSize = 4
          dtype = "float32"
        elif dataType == 'DM ':
          sampleSize = 8
          dtype = "float64"
        else:
          raise Exception(f"Expected data type FM -> float32, DM -> float64 but got {dataType}.")
        s1,rows,s2,cols = np.frombuffer(fp.read(10),dtype="int8,int32,int8,int32",count=1)[0]
        rows = int(rows)
        cols = int(cols)
        bufSize = rows * cols * sampleSize
        buf = fp.read(bufSize)
      else:
        raise Exception("Miss binary symbol before utterance in stats file.")

      data = np.frombuffer(buf,dtype=dtype).reshape([rows,cols])
      if spk == utt:
        return data
      elif spk is None:
        if result is None:
          result = data.copy()
        else:
          result += data

  if spk is not None:
    raise Exception(f"No such utterance: {spk}.")
  else:
    return result

'''A base class for CMV normalizer'''
class CMVNormalizer(ExKaldiRTBase):
  '''
  CMVN used to be embeded in FeatureProcesser.
  Note that this is not Component.
  '''
  def __init__(self,offset=-1,name=None):
    '''
    Args:
      _offset_: (int) The dim. offset.
      _name_: (str) Name.
    '''
    super().__init__(name=name)
    assert isinstance(offset,int) and offset >= -1
    self.__offset = offset

  @property
  def offset(self):
    return self.__offset

  @property
  def dim(self):
    raise Exception(f"{self.name}: Please implement the .dim function.")

class ConstantCMVNormalizer(CMVNormalizer):
  '''
  Constant CMVN.
  '''
  def __init__(self,gStats,std=False,offset=-1,name=None):
    '''
    Args:
      _gStats_: (2-d array) Previous statistics. A numpy array with shape: (2 or 1, feature dim + 1).
      _std_: (bool) If True, do variance normalization.
      _offset_: (int).
      _name_: (str).
    '''
    super().__init__(offset=offset,name=name)
    assert isinstance(std,bool), "<std> must be a bool value."
    self.__std = std
    self.redirect(gStats)

  def redirect(self,gStats):
    '''
    Redirect the global statistics.

    Args:
      _gStats_: (2-d array) Previous statistics. A numpy array with shape: (2 or 1, feature dim + 1).
    '''
    assert isinstance(gStats,np.ndarray), f"{self.name}: <gStats> of .resirect method must be a NumPy array."
    if len(gStats.shape) == 1:
      assert self.__std is False
      self.__cmv = gStats[:-1][None,:]
      self.__counter = int(gStats[-1])
    else:
      assert len(gStats.shape) == 2
      self.__cmv = gStats[:,:-1]
      self.__counter = int(gStats[0,-1])
    assert self.__counter > 0

    self.__cmvn = self.__cmv / self.__counter
    self.__dim = self.__cmvn.shape[1]
  
  @property
  def dim(self):
    '''
    Get the cmvn dim.
    '''
    return self.__dim

  def apply(self,frames):
    '''
    Apply CMVN to feature.
    If the dim of feature > the dim of cmvn, you can set offet to set cmvn range.
    '''
    if len(frames) == 0:
      return frames
    # if did not set the offet
    if self.offset == -1:
      assert frames.shape[1] == self.dim, f"{self.name}: Feature dim dose not match CMVN dim, {frames.shape[1]} != {self.dim}. "
      return ((frames - self.__cmvn[0])/self.__cmvn[1]) if self.__std else (frames - self.__cmvn[0])
    # if had offset
    else:
      endIndex = self.offset + self.dim
      assert endIndex <= frames.shape[1], f"{self.name}: cmvn dim range over flow, feature dim: {frames.shape[1]}, cmvn dim: {endIndex}."
      sliceFrames = frames[:,self.offset:endIndex]
      result = ((sliceFrames - self.__cmvn[0])/self.__cmvn[1]) if self.__std else (sliceFrames - self.__cmvn[0])
      frames[:,self.offset:endIndex] = result
      return frames

class FrameSlideCMVNormalizer(CMVNormalizer):
  '''
  Classic frame sliding CMVN.
  '''
  def __init__(self,width=600,std=False,freezedCmvn=None,gStats=None,offset=-1,dim=None,name=None):
    super().__init__(offset=offset,name=name)
    assert isinstance(width,int) and width > 0, f"{self.name}: <width> should be a reasonable value."
    assert isinstance(std,bool), f"{self.name}: <std> should be a bool value."

    self.__width = width
    self.__std = std
    self.__dim = None

    self.__freezedCmvn = None
    self.__globalCMV = None

    # If has freezed cmvn
    if freezedCmvn is not None:
      assert isinstance(freezedCmvn,np.ndarray) and len(freezedCmvn) == 2, \
             "<freezedCMVN> should be a 2-d NumPy array."
      self.__freezedCmvn = freezedCmvn
      self.__dim = freezedCmvn.shape[1]

    # If has global CMVN
    elif gStats is not None:
      assert isinstance(gStats,np.ndarray) and len(gStats) == 2, \
             "<globalCMV> should be a 2-d NumPy array."
      self.__globalCMV = gStats[:,0:-1]
      self.__globalCounter = gStats[0,-1]
      self.__dim = gStats.shape[1] - 1
      
      if self.__std:
        self.__frameBuffer = np.zeros([2,self.__width, self.__dim],dtype="float32")
        self.__cmv = np.zeros([2, self.__dim],dtype="float32")
      else:
        self.__frameBuffer = np.zeros([1,self.__width, self.__dim],dtype="float32")
        self.__cmv = np.zeros([1, self.__dim],dtype="float32")
    
    else:
      if dim is not None:
        assert isinstance(dim,int) and dim > 0
        self.__dim = dim
        if self.__std:
          self.__frameBuffer = np.zeros([2,self.__width, self.__dim],dtype="float32")
          self.__cmv = np.zeros([2, self.__dim],dtype="float32")
        else:
          self.__frameBuffer = np.zeros([1,self.__width, self.__dim],dtype="float32")
          self.__cmv = np.zeros([1, self.__dim],dtype="float32")
      else:
        self.__cmv = None
        self.__frameBuffer = None

    # Other configs
    self.__counter = 0
    self.__ringIndex = 0

  @property
  def dim(self):
    assert self.__dim is not None
    return self.__dim

  def freeze(self):
    '''Freeze the CMVN statistics.'''
    if self.__freezedCmvn is None:
      self.__freezedCmvn = self.get_cmvn()

  def apply(self,frames):
    '''Apply the cmvn to frames.'''
    assert isinstance(frames,np.ndarray)
    if len(frames) == 0:
      return frames

    assert len(frames.shape) == 2
    fdim = frames.shape[1]

    if self.offset == -1:
      # Check the feature dimmension
      if self.__dim is not None:
        assert fdim == self.dim
      # If has freezed cmvn
      if self.__freezedCmvn is not None:
        return ((frames-self.__freezedCmvn[0])/self.__freezedCmvn[1]) if self.__std else (frames-self.__freezedCmvn[0])
      else:
        return self.__apply(frames)

    else:
      # Check the feature dimmension
      if self.__dim is not None:
        assert self.offset + self.__dim <= fdim
        endIndex = self.offset + self.__dim
      else:
        self.__dim = fdim - self.offset
        endIndex = fdim
      # Compute
      sliceFrames = frames[ :, self.offset:endIndex ]
      result = self.__apply(sliceFrames)
      frames[ :, self.offset:endIndex ] = result
      return frames

  @property
  def counter(self):
    return self.__counter

  @property
  def width(self):
    return self.__width

  def __apply(self,frames):
    for ID in range(len(frames)):
      self.cache_frame(frames[ID])
      cmvn = self.get_cmvn()
      if self.__std:
        frames[ID] = (frames[ID]- cmvn[0])/cmvn[1]
      else:
        frames[ID] = (frames[ID]- cmvn[0])
    return frames

  def cache_frame(self,frame):
    '''Cache frame'''
    if self.__frameBuffer is None:
      dim = len(frame)
      if self.__std:
        self.__frameBuffer = np.zeros([2,self.__width,dim],dtype="float32")
        self.__cmv = np.zeros([2,dim],dtype="float32")

        frame2 = frame ** 2

        self.__frameBuffer[0,0,:] = frame
        self.__frameBuffer[1,0,:] = frame
        self.__cmv[0,:] = frame
        self.__cmv[1,:] = frame2

      else:
        self.__frameBuffer = np.zeros([1,self.__width,dim],dtype="float32")
        self.__cmv = np.zeros([1,dim],dtype="float32")

        self.__frameBuffer[0,0,:] = frame
        self.__cmv[0,:] = frame
      
      self.__counter = 1
      self.__ringIndex = 1
      self.__dim = dim
    else:
      self.__cmv[0] = self.__cmv[0] - self.__frameBuffer[0,self.__ringIndex,:] + frame
      self.__frameBuffer[0,self.__ringIndex,:] = frame
      if self.__std:
        frame2 = frame ** 2
        self.__cmv[1] = self.__cmv[1] - self.__frameBuffer[1,self.__ringIndex,:] + frame2
        self.__frameBuffer[1,self.__ringIndex,:] = frame2

      self.__ringIndex = (self.__ringIndex + 1)%self.__width
      self.__counter += 1

  def get_cmvn(self):
    '''Get the current statistics'''
    if self.__counter >= self.__width:
      return self.__cmv/self.__width
    else:
      if self.__globalCMV is None:
        return self.__cmv/self.__counter
      else:
        missed = self.__width - self.__counter
        if self.__globalCounter >= missed:
          return (self.__cmv + self.__globalCMV * missed/self.__globalCounter)/self.__width
        else:
          return (self.__cmv + self.__globalCMV) / (self.__counter + self.__globalCounter)
  
  def set_stats(self,stats):
    assert isinstance(stats,np.ndarray) and len(stats.shape) == 2
    self.__cmv = stats[:,0:-1]
    self.__counter = stats[0,-1]
  
  def set_freezed_cmvn(self,cmvn):
    assert isinstance(cmvn,np.ndarray) and len(cmvn.shape) == 2
    self.__freezedCmvn = cmvn

  def get_stats(self):
    '''Write the statistics into file.'''
    num = self.__counter if self.__counter < self.__width else self.__width
    return np.append(self.__cmv,[[num,],[0]],axis=1)
  
  def get_freezed_cmvn(self):
    '''Write the freezed cmvn into file.'''
    return self.__freezedCmvn

###############################################
# 3. Some functions for raw feature processing
###############################################

class MatrixFeatureProcessor(Component):
  '''
  The feature processor.
  '''
  def __init__(self,delta=0,deltaWindow=2,spliceLeft=0,spliceRight=0,
                    cmvNormalizer=None,lda=None,oKey="data",name=None):
    '''
    Args:
      _featDim_: (int) The dim. of feature.
      _batchSize_: (int) The batch size.
      _delta_: (int) The order of delta.
      _deltaWindow_: (int) The window size of delta.
      _spliceLeft_: (int) Left context to splice.
      _spliceRight_: (int) Right context to splice.
      _cmvNormalizer_: (CMVNormalizer).
      _lda_: (str, 2-d array) LDA file path or 2-d np.ndarray.
      _name_: (str) Name.
    '''
    super().__init__(oKey=oKey,name=name)
    assert isinstance(delta,int) and delta >= 0
    assert isinstance(deltaWindow,int) and deltaWindow > 0
    assert isinstance(spliceLeft,int) and spliceLeft >= 0
    assert isinstance(spliceRight,int) and spliceRight >= 0

    self.__delta = delta
    self.__deltaWindow = deltaWindow
    self.__context = ContextManager(spliceLeft,spliceRight)

    # Config LDA
    if lda is not None:
      if isinstance(lda,str):
        self.__ldaMat = load_lda_matrix(lda)
      else:
        assert isinstance(lda,np.ndarray) and len(lda.shape) == 2
        self.__ldaMat = lda
    else:
      self.__ldaMat = None
    # Config CMVNs
    self.__cmvns = []
    if cmvNormalizer is not None:
      self.set_cmvn(cmvNormalizer)

  def set_cmvn(self,cmvn,index=-1):
    assert isinstance(cmvn,CMVNormalizer),f"{self.name}: <cmvNormalizer> mush be a CMVNormalizer object but got: {type(cmvn).__name__}."
    if index == -1:
      self.__cmvns.append( cmvn )
    else:
      assert isinstance(index,int) and 0 <= index < len(self.__cmvns)
      self.__cmvns[index] = cmvn

  def __transform_function(self,feats):
    ## do the cmvn firstly.
    ## We will save the new cmvn feature instead of raw feature buffer.
    if len(self.__cmvns) > 0:
      for cmvn in self.__cmvns:
        feats = cmvn.apply( feats )

    ## then compute context 
    #print( "debug 1:", feats.shape )
    feats = self.__context.wrap( feats )
    if feats is None:
      return None
    #print( "debug 2:", feats.shape )

    # Add delta
    if self.__delta > 0: 
      feats = add_deltas(feats,order=self.__delta,window=self.__deltaWindow)
    # Splice
    if self.__context.left > 0 or self.__context.right != 0: 
      feats = splice_feats(feats,left=self.__context.left,right=self.__context.right)
    # Use LDA transform
    if self.__ldaMat is not None: 
      feats = feats.dot(self.__ldaMat)

    feats = self.__context.strip( feats )
    return feats

  def core_loop(self):

    lastPacket = None
    while True:
      
      action = self.decide_action()

      if action is False:
        break
      elif action is None:
        self.outPIPE.stop()
        break
      else:
        packet = self.get_packet()

        if is_endpoint(packet):
          self.put_packet( ENDPOINT )
        else:
          iKey = packet.mainKey if self.iKey is None else self.iKey
          newMat = self.__transform_function( packet[iKey] )
          if newMat is None:
            lastPacket = packet
            continue
          else:
            if lastPacket is None:
              packet.add( self.oKey[0], newMat, asMainKey=True )
              self.put_packet( packet )
            else:
              lastPacket.add( self.oKey[0], newMat, asMainKey=True )
              self.put_packet( lastPacket )
              lastPacket = packet
        