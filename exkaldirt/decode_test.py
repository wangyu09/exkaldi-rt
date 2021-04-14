import base
import decode
import numpy as np
import math

######################################
# Test Acoustic Estimator
######################################

def test_estimator():

  pipe = base.PIPE()

  for i in range(5):
    pipe.put( base.Packet({"data":np.ones([10,13])}, cid=i, idmaker=0) )
  
  pipe.stop()

  def test_func(frames):
    return np.zeros_like(frames)
  
  estimator = decode.AcousticEstimator(
               func=test_func,
               leftContext=3,
               rightContext=3,
               applySoftmax=False,
               applyLog=False,
            )   

  estimator.start( inPIPE=pipe )
  estimator.wait()

  print(estimator.outPIPE.size())
  print(estimator.outPIPE.get()["data"].shape)

#test_estimator()

######################################
# Test WFST Decoder
######################################

def test_decoder():

  prob = np.load("../test/1272-135031-0000_mlp.npy")

  frames = prob.shape[0]
  dim = prob.shape[1]
  batchSize = 50
  N = int(math.ceil(frames/batchSize))

  buffer = np.zeros( [ N*batchSize, dim ], dtype="float32" )
  buffer[0:frames] = prob

  pipe = base.PIPE()
  for i in range(N):
    s = i * batchSize
    e = (i+1) * batchSize
    pipe.put( base.Packet( {"data":buffer[s:e]}, cid=i, idmaker=0 ) )
  
  pipe.stop()
  
  decoder = decode.WfstDecoder(
                      symbolTable="/Work18/wangyu/kaldi/egs/mini_librispeech/s5/exp/tri3b/graph_tgsmall/words.txt",
                      silencePhones="1:2:3:4:5:6:7:8:9:10",
                      frameShiftSec=0.01,
                      tmodel="/Work18/wangyu/kaldi/egs/mini_librispeech/s5/exp/tri3b_ali_dev_clean_2/final.mdl",
                      graph="/Work18/wangyu/kaldi/egs/mini_librispeech/s5/exp/tri3b/graph_tgsmall/HCLG.fst",
                      beam=10,
                      latticeBeam=8,
                      minActive=200,
                      maxActive=7000,
                      acousticScale=0.1,
                      maxBatchSize=50,
                    )

  decoder.start(inPIPE=pipe)
  decoder.wait()

  print( decoder.outPIPE.size() )

  while not decoder.outPIPE.is_empty():
    packet = decoder.outPIPE.get()
    if base.is_endpoint(packet):
      break

    print( dict(packet.items()) )

#test_decoder()

######################################
# Test Fucntions
######################################

def test_fucntions():

  pipe = base.PIPE()

  for i in range(1,6):
    pipe.put( base.Packet({"data":"A "*i}, cid=i, idmaker=0) )
  
  pipe.put( base.ENDPOINT )

  for i in range(6,11):
    pipe.put( base.Packet({"data":"B "*i}, cid=i, idmaker=0) )
  
  pipe.stop()

  text = decode.dump_text_PIPE(pipe,endSymbol=" ")
  print( text )

test_fucntions()