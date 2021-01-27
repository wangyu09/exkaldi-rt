import numpy as np
import threading
import time
import subprocess
import os

from base import TIMEOUT, TIMESCALE, StateFlag, PIPE, Vector, Best1
from base import CMDROOT, encode_vector
from feature import apply_floor

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

def load_symbol_tab(filePath):
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

class AcousticEstimator(StateFlag):
  
  def __init__(self,featDim,chunkFrames=16,
                    padFinal=True,applySoftmax=False,applyLog=True,
                    priors=None):
    super().__init__()
    assert isinstance(featDim,int) and featDim > 0, "<featDim> must be a positive int value."
    assert isinstance(chunkFrames,int) and chunkFrames > 0, "<chunkFrames> must be a positive int value."
    assert isinstance(padFinal,bool), "<padFinal> must be a bool value."
    assert isinstance(applySoftmax,bool), "<applySoftmax> must be a bool value."
    assert isinstance(applyLog,bool), "<applyLog> must be a bool value."
    if priors is not None:
      assert isinstance(priors,np.ndarray) and len(priors.shape)==1, "<priors> must an 1-d array."

    self.__featDim = featDim
    self.__chunkFrames = chunkFrames
    self.__probabilityPIPE = PIPE()
    self.__priors = priors
    self.__memoryCache = None
    self.__pad = padFinal
    self.__applyLog = applyLog
    self.__applySoftmax = applySoftmax

    self.acoustic_function = None
    self.__featureBuffer = np.zeros([chunkFrames,featDim],dtype="float32")
    self.__dim = None

    self.__reset_position_flag()
    self.__estimateThread = None
  
  def __reset_position_flag(self):
    self.__terminationStep = False
    self.__avaliableFrames = self.__chunkFrames
    # self.set_state(None)

  @property
  def probDim(self):
    assert self.__dim is not None
    return self.__dim

  def get_memory(self):
    return self.__memoryCache

  def set_memory(self,item):
    self.__memoryCache = item

  def get_probability_pipe(self):
    return self.__probabilityPIPE

  def __prepare_chunk_feature(self,featurePIPE):
    
    timecost = 0
    pos = 0

    while pos < self.__chunkFrames:
      if featurePIPE.is_error():
        self.__set_error()
        return False
      elif featurePIPE.is_exhaustion():
        if self.__pad:
          self.__featureBuffer[pos:,:] = 0
        else:
          self.__avaliableFrames = pos
        self.__terminationStep = True
        return True
      elif featurePIPE.is_empty():
        time.sleep(TIMESCALE)
        timecost += TIMESCALE
        if timecost >= TIMEOUT:
          featurePIPE.set_error()
          self.__set_error()
          return False
      else:
        vec = featurePIPE.get()
        assert isinstance(vec,Vector)
        if pos != 0:
          self.__featureBuffer[pos] = vec.item
          pos += 1
          if vec.is_endpoint():
            self.__avaliableFrames = pos
            self.__terminationStep = True
            break
        else:
          if vec.is_endpoint():
            continue # discard this element
          self.__featureBuffer[pos] = vec.item
          pos += 1

    return True

  def __estimate_porbability(self,featurePIPE):
    
    print("Start estimating acoustic probability!")
    try:
      while True:
        # Prepare chunk feature
        self.__featureBuffer.flags.writeable = True
        if not self.__prepare_chunk_feature(featurePIPE): 
          break
        self.__featureBuffer.flags.writeable = False
        # Compute acoustic probability
        if self.__pad:
          probs = self.acoustic_function(self.__featureBuffer)
        else:
          probs = self.acoustic_function(self.__featureBuffer[:self.__avaliableFrames,:])
        self.__dim = probs.shape[-1]
        # Post-process
        if self.__applySoftmax:
          probs = softmax(probs,axis=1)
        if self.__applyLog:
          #probs = apply_floor(probs)
          probs = np.log(probs)
        if self.__priors:
          assert probs.shape[-1] == len(self.__priors), "priors dimension does not match the output of acoustic function."
          probs -= self.__priors
        # Add to PIPE
        if not self.__probabilityPIPE.is_alive():
          featurePIPE.set_error()
          self.__set_error()
          break
        for fid in range(self.__avaliableFrames-1):
          self.__probabilityPIPE.put( Vector(probs[fid],endpoint=False) )
        if self.__terminationStep:
          self.__probabilityPIPE.put( Vector(probs[self.__avaliableFrames-1],endpoint=True) )
          self.__reset_position_flag()
        else:
          self.__probabilityPIPE.put( Vector(probs[self.__avaliableFrames-1],endpoint=False) )
        # If no more data
        if featurePIPE.is_exhaustion():
          self.__set_termination()
          break
    
    except Exception as e:
      featurePIPE.set_error()
      self.__set_error()
      raise e

    finally:
      print("Stop estimating acoustic probability!")

  def start_estimating(self,featurePIPE):

    if self.acoustic_function is None:
      raise Exception("Please implement the probability computing function firstly.")
    
    self.__estimateThread = threading.Thread(target=self.__estimate_porbability, args=(featurePIPE,))
    self.__estimateThread.setDaemon(True)
    self.__estimateThread.start()

  def __set_error(self):
    self.shift_state_to_error()
    self.__probabilityPIPE.set_error()
  
  def __set_termination(self):
    self.shift_state_to_termination()
    self.__probabilityPIPE.set_termination()

  def wait(self):
    if self.__estimateThread:
      self.__estimateThread.join()

class WfstDecoder(StateFlag):

  def __init__(self,probDim,chunkFrames,symbolTable,silencePhones,frameShiftSec,tmodel,graph,
                wordBoundary=None,nBests=10,beam=16.0,maxActive=7000,minActive=200,
                latticeBeam=10.0,pruneInterval=25,
                beamDelta=0.5,hashRatio=2.0,pruneScale=0.1,
                acousticScale=0.1,lmScale=1,allowPartial=False,
                minDuration=0.1):
    super().__init__()
    assert isinstance(probDim,int) and probDim > 0, "<probDim> must be a positive int value."
    assert isinstance(chunkFrames,int) and chunkFrames > 0, "<chunkFrames> must be a positive int value."
    self.__i2wLexicon = load_symbol_tab(symbolTable)
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

    cmd = os.path.join(CMDROOT,"exkaldi-online-decoder ")
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
    cmd += f" --chunk-frames {chunkFrames} " #21
    cmd += f" --allow-partial {allowPartial} " #23
    cmd += f" --n-bests {nBests} " #25
    cmd += f" --silence-phones {silencePhones} " #27
    cmd += f" --frame-shift {frameShiftSec} " #29
    cmd += f" --tmodel {tmodel} " #31
    cmd += f" --fst {graph} " #33
    cmd += f" --word-boundary {wordBoundary} " #35
    cmd += f" --timeout { int(TIMEOUT*1000) } " #37
    cmd += f" --timescale { int(TIMESCALE*1000) } " #39

    self.__cmd = cmd
    self.__resultPIPE = PIPE()
    self.__probabilityBuffer = np.zeros([chunkFrames,probDim],dtype="float32")
    self.__chunkFrames = chunkFrames

    self.rescore_function = None

    self.__decodeProcess = None
    self.__decodeThread = None
    self.__readResultThread = None

    self.__reset_position_flag()
  
  def ids_to_words(self,IDs):
    assert isinstance(IDs,list)
    return " ".join([ self.__i2wLexicon[str(ID)] for ID in IDs ])

  def __reset_position_flag(self):
    self.__avaliableFrames = self.__chunkFrames
    self.__terminationStep = False

  def get_result_pipe(self):
    return self.__resultPIPE

  def __read_result_from_subprocess(self):
    '''
    This function is used to open a thread to read result from main decoding process. 
    '''
    counter = 0
    try:
      while True:
        line = self.__decodeProcess.stdout.readline().decode().strip()
        if line == "":
          time.sleep(TIMESCALE)
          counter += TIMESCALE
          if counter >= TIMEOUT:
            self.__set_error()
            break
        elif self.is_error() or self.__resultPIPE.is_error():
          self.__set_error()
          break
        elif self.is_termination() or self.__resultPIPE.is_termination():
          self.__decodeProcess.kill()
          self.__set_termination()
          break
        else:
          if line.startswith("-1"): # partial result
            line = line[2:].strip().split() # remove the flag "-1"
            if len(line) > 1:
              self.__resultPIPE.put( Best1( self.ids_to_words(line),endpoint=False) )
          elif line.startswith("-2"): 
            line = line[5:].strip().split("-1") # remove the flag "-2 -1"
            if self.rescore_function is None:
              #print(line[0])
              self.__resultPIPE.put( Best1( self.ids_to_words(line[0].split()),endpoint=True) ) # 
            else:
              nbestsInt = []
              for le in line:
                nbestsInt.append( [ int(ID) for ID in le.strip().split() ] )
              best1result = self.rescore_function( nbestsInt )
              self.__resultPIPE.put( Best1( self.ids_to_words(best1result),endpoint=True) )
          elif line.startswith("-3"): 
            break
          else:
            raise Exception(f"Expected flag (-1 -> partial) (-2 endpoint) (-3 termination) but got: {line}")

    except Exception as e:
      self.__set_error()
      raise e

  def __prepare_chunk_probability(self,probabilityPIPE):
    timecost = 0
    pos = 0
    while pos < self.__chunkFrames:
      if probabilityPIPE.is_error():
        self.__set_error()
        return False
      elif probabilityPIPE.is_exhaustion():
        self.__avaliableFrames = pos
        self.__terminationStep = True
        break
      elif probabilityPIPE.is_empty():
        time.sleep(TIMESCALE)
        timecost += TIMESCALE
        if timecost >= TIMEOUT:
          probabilityPIPE.set_error()
          self.__set_error()
          return False
      else:
        vec = probabilityPIPE.get()
        assert isinstance(vec,Vector)
        if pos != 0:
          self.__probabilityBuffer[pos] = vec.item
          pos += 1
          if vec.is_endpoint():
            self.__avaliableFrames = pos
            self.__terminationStep = True
            break
        else:
          if vec.is_endpoint():
            continue # discard this element
          self.__probabilityBuffer[pos] = vec.item
          pos += 1
    
    if pos == 0:
      self.__set_termination()
      return False
    else:
      return True

  def __decode(self,probabilityPIPE):
    print("Start decoding!")
    try:
      while True:
        # Prepare chunk data
        self.__probabilityBuffer.flags.writeable = True
        if not self.__prepare_chunk_probability(probabilityPIPE):
          break
        self.__probabilityBuffer *= self.__acoustic_scale
        self.__probabilityBuffer.flags.writeable = False
        # Send to decoding process
        if self.__terminationStep:
          header = f"-2 {self.__avaliableFrames} ".encode() # Partial termination
          inputs = header + encode_vector( self.__probabilityBuffer[:self.__avaliableFrames,:].reshape(-1) )
          self.__reset_position_flag()
        else:
          header = f"-1 {self.__chunkFrames} ".encode() # 
          inputs = header + encode_vector( self.__probabilityBuffer.reshape(-1) )

        try:
          self.__decodeProcess.stdin.write(inputs)
          self.__decodeProcess.stdin.flush()
        except Exception as e:
          print(self.__decodeProcess.stderr.read().decode())
          raise e
        # If no more data
        if probabilityPIPE.is_exhaustion():
          self.__decodeProcess.stdin.write(b"-3 ")
          self.__decodeProcess.stdin.flush()
          break
      # Wait until all results has been gotten. 
      while self.__readResultThread.is_alive():
        time.sleep(TIMESCALE)
      # Close the decoding process
      self.__decodeProcess.stdin.write(b"over")
      self.__set_termination()
    
    except Exception as e:
      probabilityPIPE.set_error()
      self.__set_error()
      raise e
    finally:
      print("Stop decoding!")
    
  def start_decoding(self,probabilityPIPE):
    # Open a decoding process
    self.__decodeProcess = subprocess.Popen(self.__cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    self.__decodeThread = threading.Thread(target=self.__decode,args=(probabilityPIPE,))
    self.__decodeThread.setDaemon(True)
    self.__decodeThread.start()
    # Open a thread to read result from decoding process
    self.__readResultThread = threading.Thread(target=self.__read_result_from_subprocess)
    self.__readResultThread.setDaemon(True)
    self.__readResultThread.start()

  def __set_termination(self):
    self.shift_state_to_termination()
    self.__decodeProcess.wait()
    self.__resultPIPE.set_termination()

  def __set_error(self):
    self.shift_state_to_error()
    self.__decodeProcess.stdout.close()
    self.__decodeProcess.kill()
    self.__resultPIPE.set_error()

  def wait(self):
    if self.__decodeThread:
      self.__decodeThread.join()