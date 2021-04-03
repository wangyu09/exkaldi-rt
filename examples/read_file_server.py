from exkaldirt import base,stream,transmit,feature,decode
from exkaldirt.base import info
from neural_networks import make_DNN_acoustic_model
import os

##########################
# Hyperparameters
##########################

bHostPort = 9509
rHostIP = "192.168.1.5"
rHostPort = 9510
kerasModel = "model.h5"

delta = 2
spliceLeft = 10
spliceRight = 10

featDim = (13*(delta+1)) * (spliceLeft+1+spliceRight)

##########################
# Load DNN acoustic model
##########################

KALDI_ROOT = info.KALDI_ROOT
rootDir = f"{KALDI_ROOT}/egs/mini_librispeech/s5/exp" 

words = f"{rootDir}/tri3b/graph_tgsmall/words.txt"
hmm = f"{rootDir}/tri3b_ali_train_clean_5/final.mdl"
HCLG = f"{rootDir}/tri3b/graph_tgsmall/HCLG.fst"

pdfDim = decode.get_pdf_dim(hmm)
kerasmodel = make_DNN_acoustic_model(featDim,pdfDim)
kerasmodel.load_weights(kerasModel)

##########################
# Define components
##########################

# 1. Define a receiver to receive stream from remote host.
receiver = transmit.PacketReceiver(bport=bHostPort)

receiver.decode_function = transmit.decode_value_packet

# 2. Cutter to cut frame
cutter = stream.ElementFrameCutter(width=400,shift=160)
# 3. MFCC feature extracting
extractor = feature.MfccExtractor(
                          frameDim=400, 
                          batchSize=100,
                          useEnergy=False,
                        )
# 4. processing feature
processor = feature.FeatureProcessor(
                        featDim=13,
                        batchSize=100,
                        delta=delta,
                        spliceLeft=spliceLeft,
                        spliceRight=spliceRight,
                        cmvNormalizer=feature.FrameSlideCMVNormalizer(),
                      )
# 5. acoustic probability computer
estimator = decode.AcousticEstimator(
                          featDim=featDim,
                          batchSize=100,
                          applySoftmax=True,
                          applyLog=True,
                        )

estimator.acoustic_function = lambda feats:kerasmodel(feats,training=False).numpy()

# 6. online decoder
decoder = decode.WfstDecoder(
                      probDim=pdfDim,
                      batchSize=50,
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

# 7. sender
sender = transmit.PacketSender(
        thost = rHostIP,
        tport = rHostPort,
        batchSize = 1,
      )

sender.encode_function = transmit.encode_text_packet

##########################
# Link components
##########################

chain = base.Chain()
chain.add(receiver)
chain.add(cutter)
chain.add(extractor)
chain.add(processor)
chain.add(estimator)
chain.add(decoder)
chain.add(sender)

##########################
# Run and display the results
##########################

base.dynamic_run(chain)