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

import numpy as np
import threading
import time
import subprocess
import os

from exkaldi_rt.base import info, KillableThread 
from exkaldi_rt.base import Component, PIPE, Vector, Text
from exkaldi_rt.base import encode_vector
from exkaldi_rt.feature import apply_floor
from exkaldi_rt.base import ENDPOINT, is_endpoint

def softmax(data,axis=1):
	'''
	The softmax function.

	Args:
		<data>: a Numpy array.
		<axis>: the dimension to softmax.
		
	Return:
		A new array.
	'''
  #assert isinstance(data,np.ndarray),"<data> must be NumPy array."
  #assert 0 <= axis < len(data.shape),"<axis> is out of boundary."
	if len(data.shape) == 1:
		axis = 0
	
	maxValue = data.max(axis,keepdims=True)
	dataNor = data - maxValue

	dataExp = np.exp(dataNor)
	dataExpSum = np.sum(dataExp,axis,keepdims=True)

	return dataExp / dataExpSum

def log_softmax(data,axis=1):
	'''
	The log-softmax function.

	Args:
		<data>: a Numpy array.
		<axis>: the dimension to softmax.
	Return:
		A new array.
	'''
  #assert isinstance(data,np.ndarray), "<data> must be NumPy array."
  #assert 0 <= axis < len(data.shape), "<axis> is out of boundary."

	if len(data.shape) == 1:
		axis = 0

	dataShape = list(data.shape)
	dataShape[axis] = 1
	maxValue = data.max(axis,keepdims=True)
	dataNor = data - maxValue
	
	dataExp = np.exp(dataNor)
	dataExpSum = np.sum(dataExp,axis)
	dataExpSumLog = np.log(dataExpSum) + maxValue.reshape(dataExpSum.shape)
	
	return data - dataExpSumLog.reshape(dataShape)

def load_symbol_table(filePath):
  assert os.path.isfile(filePath)
  table = {}
  with open(filePath,"r") as fr:
    lines = fr.readlines()
  for line in lines:
    w2i = line.strip().split()
    assert len(w2i) == 2
    ID = w2i[1] #int(w2i[1])
    table[ID] = w2i[0]
  
  return table

def get_pdf_dim(hmmFile):
  assert os.path.isfile(hmmFile), f"No such file: {hmmFile}."
  cmd = f"hmm-info {hmmFile} | grep pdfs"
  p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  out,err = p.communicate()
  if p.returncode != 0:
    raise Exception("Failed to get hmm info:\n" + err.decode())
  else:
    return int(out.decode().strip().split()[-1])

class AcousticEstimator(Component):
  
  def __init__(self,featDim,batchSize=16,
                    applySoftmax=False,applyLog=True,
                    priors=None,name=None):
    super().__init__(name=name)

    assert isinstance(featDim,int) and featDim > 0, "<featDim> must be a positive int value."
    assert isinstance(batchSize,int) and batchSize > 0, "<batchSize> must be a positive int value."
    assert isinstance(applySoftmax,bool), "<applySoftmax> must be a bool value."
    assert isinstance(applyLog,bool), "<applyLog> must be a bool value."
    if priors is not None:
      assert isinstance(priors,np.ndarray) and len(priors.shape)==1, "<priors> must an 1-d array."

    self.__featDim = featDim
    self.__batchSize = batchSize
    self.__priors = priors
    self.__applyLog = applyLog
    self.__applySoftmax = applySoftmax
    # The acoustic function
    self.acoustic_function = None
    # The work place
    self.__featureBuffer = np.zeros([batchSize,featDim],dtype="float32")
    # The dim of output probability
    self.__dim = None
    # Set some position flags
    self.__reset_position_flag()
    # A cache for RNN acoustic model
    self.__memoryCache = None    

  def reset(self):
    '''
    This method will be called by .reset method.
    '''
    super().reset()
    self.__featureBuffer *= 0
    self.__reset_position_flag()
    self.__dim = None
    self.__memoryCache = None

  def __reset_position_flag(self):
    self.__endpointStep = False
    self.__finalStep = False
    self.__tailIndex = self.__batchSize

  def get_prob_dim(self):
    assert self.__dim is not None
    return self.__dim

  def get_memory(self):
    return self.__memoryCache

  def set_memory(self,data):
    self.__memoryCache = data

  def __prepare_chunk_feature(self,featurePIPE):
    
    timecost = 0
    pos = 0

    while pos < self.__batchSize:
      # If feature PIPE had error
      if featurePIPE.is_wrong():
        self.kill()
        return False
      # If no more data
      elif featurePIPE.is_exhausted():
        self.__tailIndex = pos
        self.__finalStep = True
        break
      # If need wait because of receiving no data
      elif featurePIPE.is_empty():
        time.sleep(info.TIMESCALE)
        timecost += info.TIMESCALE
        if timecost > info.TIMEOUT:
          print(f"{self.name}: Timeout! Did not receive any data for a long time！")
          # Try to kill frame PIPE
          featurePIPE.kill()
          # Kill self 
          self.kill()
          return False
      # If need wait because of blocked
      elif featurePIPE.is_blocked():
        time.sleep(info.TIMESCALE)
      # If had data
      else:
        vec = featurePIPE.get()
        ## If this is an endpoint
        if is_endpoint(vec):
          self.__endpointStep = True
          self.__tailIndex = pos
          break
        else:
          assert isinstance(vec,Vector), f"{self.name}: Need Vector packet but got: {type(vec).__name__}."
          self.__featureBuffer[pos] = vec.data
          pos += 1
    # Set the rest with zero 
    self.__featureBuffer[pos:,:] = 0
    return True

  def __estimate_porbability(self,featurePIPE):
    
    print(f"{self.name}: Start...")
    try:
      while True:
        # Prepare a chunk of frames
        if not self.__prepare_chunk_feature(featurePIPE): 
          break
        # Compute acoustic probability
        if self.__tailIndex > 0:
          probs = self.acoustic_function(self.__featureBuffer[:self.__tailIndex])
          self.__dim = probs.shape[-1]
          # Post-process
          ## Softmax
          if self.__applySoftmax:
            probs = softmax(probs,axis=1)
          ## Log
          if self.__applyLog:
            probs = apply_floor(probs)
            probs = np.log(probs)
          ## Normalize with priors
          if self.__priors:
            assert probs.shape[-1] == len(self.__priors), "priors dimension does not match the output of acoustic function."
            probs -= self.__priors

        # Append to PIPE
        if self.is_wrong() or \
           self.outPIPE.is_wrong() or \
           self.outPIPE.is_terminated():
          featurePIPE.kill()
          self.kill()
          break
        else:
          ## Append to PIPE if necessary
          for i in range(self.__tailIndex):
            self.outPIPE.put( Vector(probs[i]) )
          ## If arrived ENDPOINT
          if self.__endpointStep:
            self.outPIPE.put( ENDPOINT )
            self.__reset_position_flag()
          ## If over
          if self.__finalStep or self.is_terminated():
            self.stop()
            break
    except Exception as e:
      featurePIPE.kill()
      self.kill()
      raise e
    finally:
      print(f"{self.name}: Stop!")

  def _start(self,inPIPE):

    if self.acoustic_function is None:
      raise Exception(f"{self.name}: Please implement the probability computing function firstly.")
    
    estimateThread = KillableThread(target=self.__estimate_porbability, args=(inPIPE,))
    estimateThread.setDaemon(True)
    estimateThread.start()

    return estimateThread

class WfstDecoder(Component):

  def __init__(self,probDim,batchSize,symbolTable,silencePhones,frameShiftSec,tmodel,graph,
                wordBoundary=None,nBests=10,beam=16.0,maxActive=7000,minActive=200,
                latticeBeam=10.0,pruneInterval=25,
                beamDelta=0.5,hashRatio=2.0,pruneScale=0.1,
                acousticScale=0.1,lmScale=1,allowPartial=False,
                minDuration=0.1,name=None):
    super().__init__(name=name)
    assert isinstance(probDim,int) and probDim > 0, "<probDim> must be a positive int value."
    assert isinstance(batchSize,int) and batchSize > 0, "<batchSize> must be a positive int value."
    self.__i2wLexicon = load_symbol_table(symbolTable)
    assert isinstance(silencePhones,str) # check the format
    assert isinstance(frameShiftSec,float) and frameShiftSec > 0, "<silencePhones> must be a positive float value."
    assert os.path.isfile(tmodel), "<tmodel> should be a file path."
    assert os.path.isfile(graph), "<graph> should be a file path."
    if wordBoundary is not None:
      assert os.path.isfile(wordBoundary), "<wordBoundary> should be a file path."
    assert isinstance(nBests,int) and nBests > 1, "<nBests> must be an int value and greater than 1."
    assert isinstance(beam,(int,float)) and beam > 0, "<beam> must be a positive float value."
    assert isinstance(maxActive,int) and maxActive > 0, "<maxActive> must be a positive int value."
    assert isinstance(minActive,int) and minActive > 0, "<minActive> must be a positive int value."
    assert maxActive > minActive
    assert isinstance(latticeBeam,(int,float)) and latticeBeam > 0, "<latticeBeam> must be a positive float value."
    assert isinstance(pruneInterval,int) and pruneInterval > 0, "<pruneInterval> must be a positive int value."
    assert isinstance(beamDelta,(int,float)) and beamDelta > 0, "<beamDelta> must be a positive float value."
    assert isinstance(hashRatio,(int,float)) and hashRatio > 0, "<hashRatio> must be a positive float value."
    assert isinstance(pruneScale,(int,float)) and pruneScale > 0, "<pruneScale> must be a positive float value."
    assert isinstance(acousticScale,(int,float)) and acousticScale > 0, "<acousticScale> must be a positive float value."
    assert isinstance(lmScale,(int,float)) and lmScale > 0, "<lmScale> must be a positive float value."
    assert isinstance(allowPartial,bool), "<allowPartial> must be a bool value."
    assert isinstance(minDuration,(int,float)) and minDuration > 0, "<minDuration> must be a positive float value."

    self.__acoustic_scale = acousticScale

    # Config the subprocess command
    cmd = os.path.join(info.CMDROOT,"exkaldi-online-decoder ")
    cmd += f" --beam {beam} " #1
    cmd += f" --max-active {maxActive} " #3
    cmd += f" --min-active {minActive} " #5
    cmd += f" --lattice-beam {latticeBeam} " #7
    cmd += f" --prune-interval {pruneInterval} " #9
    cmd += f" --beam-delta {beamDelta} " #11
    cmd += f" --hash-ratio {hashRatio} " #13
    cmd += f" --prune-scale {pruneScale} " #15
    cmd += f" --acoustic-scale {acousticScale} " #17
    cmd += f" --lm-scale {lmScale} " #19
    cmd += f" --chunk-frames {batchSize} " #21
    cmd += f" --allow-partial {allowPartial} " #23
    cmd += f" --n-bests {nBests} " #25
    cmd += f" --silence-phones {silencePhones} " #27
    cmd += f" --frame-shift {frameShiftSec} " #29
    cmd += f" --tmodel {tmodel} " #31
    cmd += f" --fst {graph} " #33
    cmd += f" --word-boundary {wordBoundary} " #35
    cmd += f" --timeout { int(info.TIMEOUT*1000) } " #37
    cmd += f" --timescale { int(info.TIMESCALE*1000) } " #39
    self.__cmd = cmd
    # Check the dim of probability
    pdfs = get_pdf_dim(tmodel)
    assert probDim == pdfs, "The dimension of probability does not match the pdfs of hmm model. " + \
                            "You might use a different hmm model. "

    # The work place
    self.__probabilityBuffer = np.zeros([batchSize,probDim],dtype="float32")
    self.__batchSize = batchSize
    # A rescoring fucntions
    self.rescore_function = None
    # The main subprocess to run the decoding loop
    self.__decodeProcess = None
    # A thread to read results from decoding subprocess
    self.__readResultThread = None
    # Reset some position flags.
    self.__reset_position_flag()

  def reset(self):
    super().reset()
    self.__probabilityBuffer *= 0
    self.__decodeProcess = None
    self.__readResultThread = None
    self.__reset_position_flag()

  def ids_to_words(self,IDs):
    assert isinstance(IDs,list)
    result = []
    for ID in IDs:
      ID = str(ID)
      if ID in self.__i2wLexicon.keys():
        result.append( self.__i2wLexicon[ID] )
      else:
        result.append( "<UNK>" )

    return " ".join(result)

  def __reset_position_flag(self):
    self.__tailIndex = self.__batchSize
    self.__finalStep = False
    self.__endpointStep = False

  def __read_result_from_subprocess(self):
    '''
    This function is used to open a thread to read result from main decoding process. 
    '''
    timecost = 0
    try:
      while True:
        # Read
        line = self.__decodeProcess.stdout.readline().decode().strip()
        # Readed nothing
        if line == "":
          time.sleep(info.TIMESCALE)
          timecost += info.TIMESCALE
          if timecost > info.TIMEOUT:
            print(f"{self.name}: Timeout! Receiving thread has not received any data for a long time！")
            self.kill()
            break
        # If error occurred
        elif self.is_wrong() or \
             self.outPIPE.is_wrong() or \
             self.outPIPE.is_terminated():
            self.kill()
            break

        else:
          #print("result:",line)
          ## partial result
          if line.startswith("-1"):
            line = line[2:].strip().split() # discard the flag "-1"
            if len(line) > 0:
              #print(line, self.ids_to_words(line))
              #self.outPIPE.put( Text( " ".join(line) ) )
              self.outPIPE.put( Text( self.ids_to_words(line) ) )

          ## Endpoint
          elif line.startswith("-2"): 
            line = line[5:].strip().split("-1") # discard the flag "-2 -1"
            if self.rescore_function is None:
              line = line[0].strip()
              if len(line) > 0:
                self.outPIPE.put( Text( self.ids_to_words(line.split()) ) )
            else:
              nbestsInt = []
              for le in line:
                le = le.strip()
                if len(le) > 0:
                  nbestsInt.append( [ int(ID) for ID in le.split() ] )
              if len(nbestsInt) > 0:
                best1result = self.rescore_function( nbestsInt )
                self.outPIPE.put( Text( self.ids_to_words(best1result) ) )
            ### Add a endpoint flag
            self.outPIPE.put( ENDPOINT )

          ## Final step
          elif line.startswith("-3") or self.is_terminated(): 
            #self.outPIPE.put( ENDPOINT )
            #!!!! do not stop this componnet
            break

          else:
            raise Exception(f"{self.name}: Expected flag (-1 -> partial) (-2 endpoint) (-3 termination) but got: {line}")

    except Exception as e:
      self.kill()
      raise e

  def __prepare_chunk_probability(self,probabilityPIPE):

    timecost = 0
    pos = 0
    
    while pos < self.__batchSize:
      # If the previous PIPE had errors
      if probabilityPIPE.is_wrong():
        self.kill()
        return False
      # If no more data
      elif probabilityPIPE.is_exhausted():
        self.__tailIndex = pos
        self.__finalStep = True
        break
      # If need wait because of receiving no data
      elif probabilityPIPE.is_empty():
        time.sleep(info.TIMESCALE)
        timecost += info.TIMESCALE
        if timecost > info.TIMEOUT:
          print(f"{self.name}: Timeout! Did not receive any data for a long time！")
          # Try to kill frame PIPE
          probabilityPIPE.kill()
          # Kill self 
          self.kill()
          return False
      # If need wait because of blocked
      elif probabilityPIPE.is_blocked():
        time.sleep(info.TIMESCALE)
      # If had data
      else:
        vec = probabilityPIPE.get()
        if is_endpoint(vec):
          self.__endpointStep = True
          self.__tailIndex = pos
          break
        else:
          assert isinstance(vec,Vector)
          self.__probabilityBuffer[pos] = vec.data
          pos += 1
    # pad the rest
    self.__probabilityBuffer[pos:,:] = 0
    return True

  def __decode(self,probabilityPIPE):
    print(f"{self.name}: Start...")
    try:
      while True:
        # Prepare a chunk of frames
        if not self.__prepare_chunk_probability(probabilityPIPE):
          break
        self.__probabilityBuffer *= self.__acoustic_scale
        try:
          # Send to decoding process
          if self.__tailIndex > 0:
            ## "-1" is activity flag 
            header = f"-1 {self.__tailIndex} ".encode()
            inputs = header + encode_vector( self.__probabilityBuffer[:self.__tailIndex,:].reshape(-1) )
            self.__decodeProcess.stdin.write(inputs)
            self.__decodeProcess.stdin.flush()
  
          # If this is an endpoint step
          if self.__endpointStep:
            self.__decodeProcess.stdin.write(b"-2 ")
            self.__decodeProcess.stdin.flush()
            self.__reset_position_flag()

          # If this is final step
          elif self.__finalStep:
            self.__decodeProcess.stdin.write(b"-3 ")
            self.__decodeProcess.stdin.flush()
            #!!!! do not stop this Component
            break

        except Exception as e:
          print(self.__decodeProcess.stderr.read().decode())
          raise e

        if self.is_terminated():
          break

      # Wait until all results has been gotten. 
      self.__readResultThread.join()
      # Close the decoding process
      self.__decodeProcess.stdin.write(b"over")
      self.stop()
    
    except Exception as e:
      probabilityPIPE.kill()
      self.kill()
      raise e
    finally:
      print(f"{self.name}: Stop!")
    
  def _start(self,inPIPE):
    # Open a decoding process
    self.__decodeProcess = subprocess.Popen(self.__cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    decodeThread = KillableThread(target=self.__decode,args=(inPIPE,))
    decodeThread.setDaemon(True)
    decodeThread.start()
    # Open a thread to read result from decoding process
    self.__readResultThread = KillableThread(target=self.__read_result_from_subprocess)
    self.__readResultThread.setDaemon(True)
    self.__readResultThread.start()

    return decodeThread

  # rewrite the stop function
  def stop(self):
    super().stop()
    self.__decodeProcess.stdout.close()
    self.__decodeProcess.kill()

  # rewrite the kill function
  def kill(self):
    super().kill()
    self.__decodeProcess.stdout.close()
    self.__decodeProcess.kill()

def dump_text_PIPE(textPIPE,allowPartial=True,endSymbol="\n"):
  '''
  Dump a text PIPE to a transcription.
  '''
  assert isinstance(allowPartial,bool)
  assert isinstance(endSymbol,str)
  assert textPIPE.is_alive() or textPIPE.is_terminated(), "<textPIPE> must be ALIVE or TERMINATION PIPE."
  
  result = []
  memory = None
  timecost = 0

  while True:
    if textPIPE.is_wrong() or textPIPE.is_exhausted():
      break
    elif textPIPE.is_empty():
      time.sleep(info.TIMESCALE)
      timecost += info.TIMESCALE
      if timecost > info.TIMEOUT:
        break

    else:
      packet = textPIPE.get()
      if is_endpoint(packet):
        if memory is None:
          continue
        else:
          result.append( memory )
          memory = None
      else:
        assert isinstance(packet,Text), "This is not a Text PIPE."
        memory = packet.data
        #print(memory)

  if allowPartial and (memory is not None):
    result.append( memory )

  return endSymbol.join(result)