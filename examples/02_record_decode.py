from exkaldirt import base,stream,feature,decode
from exkaldirt.base import info
from neural_networks import make_DNN_acoustic_model
import time

##########################
# Hyperparameters
##########################

kerasModel = "model.h5"
words = "words.txt"
hmm = "final.mdl"
HCLG = "HCLG.fst"

delta = 2
spliceLeft = 10
spliceRight = 10

featDim = (13*(delta+1)) * (spliceLeft+1+spliceRight)

##########################
# Load DNN acoustic model
##########################

pdfDim = decode.get_pdf_dim(hmm)
kerasmodel = make_DNN_acoustic_model(featDim,pdfDim)
kerasmodel.load_weights(kerasModel)

##########################
# Define components
##########################

# 1. Create a stream recorder to read realtime stream from microphone
recorder = stream.StreamRecorder()
# 2. Cutter to cut frame
cutter = stream.ElementFrameCutter(batchSize=50,width=400,shift=160)
# 3. MFCC feature extracting
extractor = feature.MfccExtractor(
                          useEnergy=False,
                        )
# 4. processing feature
processor = feature.MatrixFeatureProcessor(
                        delta=delta,
                        spliceLeft=spliceLeft,
                        spliceRight=spliceRight,
                        cmvNormalizer=feature.FrameSlideCMVNormalizer(),
                      )
# 5. acoustic probability computer
def keras_compute(feats):
  return kerasmodel(feats,training=False).numpy()

estimator = decode.AcousticEstimator(
                          keras_compute,
                          applySoftmax=True,
                          applyLog=True,
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
# Link components
##########################

chain = base.Chain()
chain.add(recorder)
chain.add(cutter)
chain.add(extractor)
chain.add(processor)
chain.add(estimator)
chain.add(decoder)

##########################
# Run and display the results
##########################

chain.start()

base.dynamic_display(chain.outPIPE, mapFunc=lambda packet:print("Result >> ",packet[packet.mainKey]))