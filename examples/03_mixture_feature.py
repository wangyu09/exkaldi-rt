from exkaldirt import base,stream,feature,decode,joint
from exkaldirt.base import info

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import glob
import sys
from neural_networks import make_DNN_acoustic_model

##########################
# Hyperparamaters
##########################

wavFile = "../examples/84-121550-0000.wav"

fbankDelta = 0
fbankContext = 5

mfccDelta = 2
mfccContext = 5

ldaDelta = 0
ldaContext = 5

kerasModel = "/Work19/wangyu/exkaldi-rt/experiments/feat/mixture/mixed_exp/out_20210305-000839/model_ep22.h5"

##########################
# other parameters
##########################

featDim = 23*(fbankDelta+1)*(fbankContext+1) + 13*(mfccDelta+1)*(mfccContext+1) + 40*(ldaDelta+1)*(ldaContext+1)

utt2spk= f"{info.KALDI_ROOT}/egs/mini_librispeech/s5/data/dev_clean_2/utt2spk"
fbankCmvn = f"{info.KALDI_ROOT}/egs/mini_librispeech/s5/fbank/cmvn_dev_clean_2.ark"
mfccCmvn = f"{info.KALDI_ROOT}/egs/mini_librispeech/s5/mfcc/cmvn_dev_clean_2.ark"
ldaCmvn = f"{info.KALDI_ROOT}/egs/mini_librispeech/s5/lda/cmvn_dev_clean_2.ark"
ldaFile = f"{info.KALDI_ROOT}/egs/mini_librispeech/s5/exp/tri3b/final.mat"
words = f"{info.KALDI_ROOT}/egs/mini_librispeech/s5/exp/tri3b/graph_tgsmall/words.txt"
hmm = f"{info.KALDI_ROOT}/egs/mini_librispeech/s5/exp/tri3b_ali_dev_clean_2/final.mdl"
HCLG = f"{info.KALDI_ROOT}/egs/mini_librispeech/s5/exp/tri3b/graph_tgsmall/HCLG.fst"

uttID = os.path.basename(wavFile).split()[0]
spkID = feature.utt_to_spk(uttID,utt2spk=utt2spk)

globalCMVNFbank = feature.get_kaldi_cmvn(fbankCmvn, spk=spkID)
globalCMVNMfcc = feature.get_kaldi_cmvn(mfccCmvn, spk=spkID)
globalCMVNLda = feature.get_kaldi_cmvn(ldaCmvn, spk=spkID)

pdfDim = decode.get_pdf_dim(hmm)
kerasmodel = make_DNN_acoustic_model(featDim,pdfDim) # MFCC 429 ( 13+d+dd * 11 ) + Fbank 253 ( 23 * 11 ) + LDA 440 ( 40 * 11 )
kerasmodel.load_weights(kerasModel)

########################
# Define components
########################

# 1. Create a stream reader to read realtime stream from audio file
reader = stream.StreamReader(wavFile,simulate=False)#,vaDetector=stream.WebrtcVADetector(mode=3))
# 2. Cutter to cut frame
cutter = stream.ElementFrameCutter(batchSize=50,width=400,shift=160)
# 3. Mixture feature extracting
extractor = feature.MixtureExtractor(
                        mixType=["mfcc","fbank"],
                        useEnergyForFbank=False,
                        useEnergyForMfcc=False,
                        minParallelSize=100,
                      )
# 4. joint
def split_func(item):
  return {"fbank":item["fbank"]}, {"mfcc":item["mfcc"]}, {"lda":item["mfcc"]}
spliter = joint.Spliter(split_func,outNums=3)
# 5. processor for fbank feature
processorFbank = feature.MatrixFeatureProcessor(
                            delta=fbankDelta,
                            spliceLeft=fbankContext,
                            spliceRight=fbankContext,
                            cmvNormalizer=feature.ConstantCMVNormalizer(gStats=globalCMVNFbank),
                            oKey="fbank",
                          )
# 6. processor for mfcc feature
processorMfcc = feature.MatrixFeatureProcessor(
                            delta=mfccDelta,
                            spliceLeft=mfccContext,
                            spliceRight=mfccContext,
                            cmvNormalizer=feature.ConstantCMVNormalizer(gStats=globalCMVNMfcc),
                            oKey="mfcc",
                          )
# 7. processor for lda feature
processorLdaPre = feature.MatrixFeatureProcessor(
                            delta=0,
                            spliceLeft=3,
                            spliceRight=3,
                            cmvNormalizer=feature.ConstantCMVNormalizer(gStats=globalCMVNMfcc),
                            lda=ldaFile,
                            oKey="lda",
                          )
processorLda = feature.MatrixFeatureProcessor(
                            delta=ldaDelta,
                            spliceLeft=ldaContext,
                            spliceRight=ldaContext,
                            cmvNormalizer=feature.ConstantCMVNormalizer(gStats=globalCMVNLda),
                            oKey="lda",
                          )
# 8. combine feature
def combine_func(items):
  fbank = items[0]["fbank"]
  mfcc = items[1]["mfcc"]
  lda = items[2]["lda"]
  return { "data": np.concatenate([mfcc,fbank,lda],axis=1)  }

combiner = joint.Combiner(combine_func)

# acoustic function

def keras_compute(feats):
  global kerasmodel
  predPdf = kerasmodel(feats, training=False)
  return predPdf.numpy()

estimator = decode.AcousticEstimator(
                          keras_compute,
                          applySoftmax=True,
                          applyLog=True,
                          priors=None
                        )
                   
# 6. online decoder
decoder = decode.WfstDecoder(
                      symbolTable=words,
                      silencePhones="1:2:3:4:5:6:7:8:9:10",
                      frameShiftSec=160/16000,
                      tmodel=hmm,
                      graph=HCLG,
                      beam=10,
                      latticeBeam=8,
                      minActive=200,
                      maxActive=7000,
                      acousticScale=0.1
                    )

##########################
# Step 2: Link this components
##########################

chain = base.Chain()
chain.add( reader )
chain.add( cutter )
chain.add( extractor )
chain.add( spliter ) 
chain.add( processorFbank, inPIPE=spliter.outPIPE[0] )
chain.add( processorMfcc, inPIPE=spliter.outPIPE[1] )
chain.add( processorLdaPre, inPIPE=spliter.outPIPE[2] )
chain.add( processorLda, inPIPE=processorLdaPre.outPIPE )
chain.add( combiner, inPIPE=[processorFbank.outPIPE,processorMfcc.outPIPE,processorLda.outPIPE] )
chain.add( estimator )
chain.add( decoder )

chain.start()
chain.wait()

print( chain.outPIPE )
packet = chain.outPIPE.get()
print( packet.keys() )

result = decode.dump_text_PIPE(chain.outPIPE, endSymbol=" ")
print(result)

st = reader.outPIPE.report_time().firstGet
st1 = reader.outPIPE.report_time().lastGet
et = decoder.outPIPE.report_time().lastPut

wav = stream.read(wavFile)

print("Duration:",wav.duration)
print("Time Cost:",(et-st).total_seconds())
print("RTF:", (et-st).total_seconds() / wav.duration )
print("Latency:", (et-st1).total_seconds() )
