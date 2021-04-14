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

import numpy as np
import threading
import time
import subprocess
import os
import queue

#from exkaldirt.base import info 
#from exkaldirt.base import Component, PIPE, Packet, ContextManager
#from exkaldirt.utils import encode_vector_temp
#from exkaldirt.feature import apply_floor
#from exkaldirt.base import ENDPOINT, is_endpoint, print_

from base import info, mark
from base import Component, PIPE, Packet, ContextManager
from utils import encode_vector_temp
from feature import apply_floor
from base import ENDPOINT, is_endpoint, print_

def softmax(data,axis=1):
  assert isinstance(data,np.ndarray)
  if len(data.shape) == 1:
    axis = 0
  else:
    assert 0 <= axis < len(data.shape)
  maxValue = data.max(axis,keepdims=True)
  dataNor = data - maxValue
  dataExp = np.exp(dataNor)
  dataExpSum = np.sum(dataExp,axis,keepdims=True)
  return dataExp / dataExpSum

def log_softmax(data,axis=1):

  assert isinstance(data,np.ndarray)
  if len(data.shape) == 1:
    axis = 0
  else:
    assert 0 <= axis < len(data.shape)

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
  
  def __init__(self,func,leftContext=0,rightContext=0,applySoftmax=False,applyLog=True,
                    priors=None,oKey="data",name=None):
    super().__init__(name=name)
    assert isinstance(applySoftmax,bool), "<applySoftmax> must be a bool value."
    assert isinstance(applyLog,bool), "<applyLog> must be a bool value."
    if priors is not None:
      assert isinstance(priors,np.ndarray) and len(priors.shape)==1, "<priors> must an 1-d array."

    self.__priors = priors
    self.__applyLog = applyLog
    self.__applySoftmax = applySoftmax
    # The acoustic function
    assert callable(func)
    self.acoustic_function = func
    self.__memoryCache = None   
    assert isinstance(leftContext,int) and leftContext >= 0
    assert isinstance(rightContext,int) and rightContext >= 0
    if leftContext > 0 or rightContext > 0:
      self.__context = ContextManager(left=leftContext,right=rightContext)
    else:
      self.__context = None

  def get_memory(self):
    return self.__memoryCache

  def set_memory(self,data):
    self.__memoryCache = data

  def core_loop(self):
    
    lastPacket = None
    firstComputing = True
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
          mat = packet[iKey]
          if self.__context is not None:
            mat = self.__context.wrap( mat )
            if mat is None:
              lastPacket = packet
              continue
            
          probs = self.acoustic_function( mat )
          assert isinstance(probs,np.ndarray) and len(probs.shape) == 2
          if len(probs) != packet[iKey].shape[0] and firstComputing:
            print_( f"Warning! {self.name}: The number of frames has changed. Please make sure this is indeed the result you want." )
            firstComputing = False
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

          if lastPacket is None:
            packet.add( self.oKey[0], probs, asMainKey=True )
            self.put_packet( packet )
          else:
            lastPacket.add( self.oKey[0], probs, asMainKey=True )
            self.put_packet( lastPacket )
            lastPacket = packet

class WfstDecoder(Component):

  def __init__(self,symbolTable,silencePhones,frameShiftSec,tmodel,graph,
                wordBoundary=None,nBests=10,beam=16.0,maxActive=7000,minActive=200,
                latticeBeam=10.0,pruneInterval=25,
                beamDelta=0.5,hashRatio=2.0,pruneScale=0.1,
                acousticScale=0.1,lmScale=1,allowPartial=False,
                minDuration=0.1,oKey="data",maxBatchSize=100,name=None):

    super().__init__(oKey=oKey,name=name)
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
    assert isinstance(maxBatchSize,int) and maxBatchSize > 1
    self.__max_batch_size = maxBatchSize

    # Config the subprocess command
    cmd = os.path.join( info.CMDROOT,"exkaldi-online-decoder ")
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
    cmd += f" --chunk-frames {maxBatchSize} " #21
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
    self.__pdfs = get_pdf_dim(tmodel)

    # A rescoring fucntions
    self.rescore_function = None
    # The main subprocess to run the decoding loop
    self.__decodeProcess = None
    # A thread to read results from decoding subprocess
    self.__readResultThread = None
    # id
    self.__packetCache = queue.Queue()

  def reset(self):
    super().reset()
    self.__decodeProcess = None
    self.__readResultThread = None

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

  def __read_result_from_subprocess(self):
    '''
    This function is used to open a thread to read result from main decoding process. 
    '''
    timecost = 0
    try:
      while True:
        # decide state and action
        master, state = self.decide_state()

        if state == mark.wrong:
          break
        elif state == mark.stranded:
          time.sleep( info.TIMESCALE )
          continue
        elif state == mark.terminated:
          if master == mark.outPIPE:
            break
        
        # if state is active or terminated (master is inPIPE)
        # do the following steps

        # Read
        line = self.__decodeProcess.stdout.readline().decode().strip()

        # nothing is received
        if line == "":
          time.sleep(info.TIMESCALE)
          timecost += info.TIMESCALE
          if timecost > info.TIMEOUT:
            raise Exception(f"{self.name}: Timeout! Receiving thread has not received any data for a long time！")

        else:

          if line.startswith("-1"):
            # pick out a cached picket
            packet = self.__packetCache.get()
            # 
            line = line[2:].strip().split() # discard the flag "-1"
            if len(line) > 0:
              packet.add( self.oKey[0], self.ids_to_words(line), asMainKey=True )
            else:
              packet.add( self.oKey[0], " ", asMainKey=True )
            self.put_packet( packet )

          ## Endpoint
          elif line.startswith("-2"): 
            # pick out a cached picket
            packet = self.__packetCache.get()
            #
            lines = line[5:].strip().split("-1") # discard the flag "-2 -1"
            lines = [ line.strip().split() for line in lines if len(line.strip()) > 0 ] 
            if len(lines) == 0:
              packet.add( self.oKey[0], " ", asMainKey=True )
            elif len(lines) == 1:
              packet.add( self.oKey[0], self.ids_to_words(lines[0]), asMainKey=True )
            else:
              # do not need to rescore
              if self.rescore_function is None:
                for i, line in enumerate(lines):
                  outKey = self.oKey[0] if i == 0 else ( self.oKey[0] + f"-{i+1}" )
                  packet.add( outKey, self.ids_to_words(line), asMainKey=True )

              else:
                nbestsInt = [ [ int(ID) for ID in line.split() ] for line in lines ]
                nResults = self.rescore_function( nbestsInt )
                assert isinstance(nbestsInt,(list,tuple)) and len(nbestsInt) > 0
                for i,re in enumerate(nResults):
                  assert isinstance(re,(list,tuple)) and len(nbestsInt) > 0
                  outKey = self.oKey[0] if i == 0 else ( self.oKey[0] + f"-{i+1}" )
                  packet.add( outKey, self.ids_to_words(re), asMainKey=True )
              
              self.put_packet( packet )
            
            ### Add a endpoint flag
            self.put_packet( ENDPOINT )

          ## Final step
          elif line.startswith("-3"): 
            break

          else:
            raise Exception(f"{self.name}: Expected flag (-1 -> partial) (-2 endpoint) (-3 termination) but got: {line}")

    except Exception as e:
      if not self.inPIPE.state_is_(mark.wrong,mark.terminated):
        self.inPIPE.kill()
      if not self.inPIPE.state_is_(mark.wrong,mark.terminated):
        self.inPIPE.kill()
      raise e
    else:
      if not self.inPIPE.state_is_(mark.wrong,mark.terminated):
        self.inPIPE.terminate()
      if not self.inPIPE.state_is_(mark.wrong,mark.terminated):
        self.inPIPE.terminate()
    finally:
      self.__decodeProcess.stdout.close()
      self.__decodeProcess.kill()

  def core_loop(self):
    
    # open exkaldi online decoding process
    self.__decodeProcess = subprocess.Popen(self.__cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    # open reading result thread
    self.__readResultThread = threading.Thread(target=self.__read_result_from_subprocess)
    self.__readResultThread.setDaemon(True)
    self.__readResultThread.start()
    
    # start core loop
    try:
      while True:
        action = self.decide_action()

        if action is False:
          break
        elif action is None:
          # final step
          try:
            self.__decodeProcess.stdin.write(b"-3 ")
            self.__decodeProcess.stdin.flush()
          except Exception as e:
            print(self.__decodeProcess.stderr.read().decode())
            raise e
          break

        else:
          packet = self.get_packet()
          if is_endpoint(packet):
            try:
              self.__decodeProcess.stdin.write(b"-2 ")
              self.__decodeProcess.stdin.flush()
            except Exception as e:
              print(self.__decodeProcess.stderr.read().decode())
              raise e  
          else:
            iKey = packet.mainKey if self.iKey is None else self.iKey
            self.__packetCache.put( packet )
            mat = self.__acoustic_scale * packet[iKey]

            assert mat.shape[0] <= self.__max_batch_size, "The chunk size of matrix > max allowable batch size of this decoder."
            assert mat.shape[1] == self.__pdfs, "The dim. of probability does not match the PDFs."

            header = f"-1 {mat.shape[0]} ".encode()
            inputs = header + encode_vector_temp( mat.reshape(-1) )
            try:
              self.__decodeProcess.stdin.write(inputs)
              self.__decodeProcess.stdin.flush()
            except Exception as e:
              print(self.__decodeProcess.stderr.read().decode())
              raise e

      # Wait until all results has been gotten. 
      self.__readResultThread.join()
      # Close the decoding process
      self.__decodeProcess.stdin.write(b"over")
    finally:
      self.__decodeProcess.stdout.close()
      self.__decodeProcess.kill()

def dump_text_PIPE(pipe,key=None,allowPartial=True,endSymbol="\n"):
  '''
  Dump a text PIPE to a transcription.
  '''
  assert isinstance(allowPartial,bool)
  assert isinstance(endSymbol,str)
  assert pipe.state_is_(mark.wrong,mark.terminated), "<pipe> must be wrong or terminated PIPE."
  assert not pipe.is_outlocked()
  if key is not None:
    assert isinstance(key,str)
  
  result = []
  memory = None

  while True:
    if pipe.is_empty():
      break
    else:
      packet = pipe.get()
      if is_endpoint(packet):
        if memory is None:
          continue
        else:
          result.append( memory )
          memory = None
      else:
        iKey = packet.mainKey if key is None else key
        text = packet[iKey]
        assert isinstance(text,str)
        memory = text

  if allowPartial and (memory is not None):
    result.append( memory )

  return endSymbol.join(result)