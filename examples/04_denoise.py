from exkaldirt import base,stream,feature,decode
from exkaldirt.base import info
from neural_networks import make_DNN_acoustic_model, load_denoise_model
import os
import numpy as np

##########################
# Hyperparameters
##########################

waveFile = "noise_1272-135031-0003.wav"
kerasModel = "model.h5"
denoiseModelFile = "saved_model"

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

denoiseModel = load_denoise_model( denoiseModelFile )

##########################
# Define components
##########################

# 1. Create a stream reader to read realtime stream from audio file
reader = stream.StreamReader(waveFile,simulate=True)
# 2. Cutter to cut frame
cutter = stream.ElementFrameCutter(batchSize=64,width=400,shift=160)
# 3. MFCC feature extracting with denoising

def denoise_from_magnitude(frames,width=32):
  global denoiseModel
  originLen = frames.shape[0]
  magnitude, phase = frames[:,:,0].T, frames[:,:,1].T
  dim = magnitude.shape[0]
  # magnitude
  magnitude = np.abs( magnitude  )
  magnitude /= magnitude.max()
  if originLen < width:
    magnitude = np.concatenate([magnitude,np.zeros([dim,width-originLen])],axis=1)
  magnitude = denoiseModel.predict( magnitude[None,:,:,None] )[0,:,:originLen,:]
  # phase
  phase = np.abs( phase  )
  phase /= phase.max()
  if originLen < width:
    phase = np.concatenate([phase,np.zeros([dim,width-originLen])],axis=1)
  phase = denoiseModel.predict( phase[None,:,:,None] )[0,:,:originLen,:]
  # concat
  result = np.concatenate([magnitude,phase],axis=2)
  return result.transpose([1,0,2])

windows = feature.get_window_function(size=400,winType="povey")
fftLen = feature.get_padded_fft_length(400)
melFilters = feature.get_mel_bins(numBins=23,rate=16000,fftLen=fftLen,lowFreq=20,highFreq=0)
dctMat = feature.get_dct_matrix(numCeps=13,numBins=23)
cepsCoeff = feature.get_cepstral_lifter_coeff(dim=13,factor=22)

def denoise_extract(frames):
  global windows,melFilters,dctMat,cepsCoeff
  # dither
  frames = feature.dither_singal_2d(frames, 1.0)
  # remove dc
  frames = feature.remove_dc_offset_2d(frames)
  # preemphasize
  frames = feature.pre_emphasize_2d(frames, 0.97)
  frames *= windows
  # Compute energy
  energies = feature.compute_log_energy_2d(frames)
  # do FFT
  _, frames = feature.split_radix_real_fft_2d(frames)
  # do denoise
  frames = denoise_from_magnitude(frames,32)
  # power
  frames = feature.compute_power_spectrum_2d(frames)
  # Mel filtering
  frames = np.dot( frames, melFilters )
  frames = feature.apply_floor(frames)
  frames = np.log(frames)
  # Dct
  frames = frames.dot(dctMat)
  frames = frames * cepsCoeff
  # Add energy
  frames[:,0] = energies

  return frames

extractor = feature.MatrixFeatureExtractor(
                          extFunc=denoise_extract,
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
chain.add(reader)
chain.add(cutter)
chain.add(extractor)
chain.add(processor)
chain.add(estimator)
chain.add(decoder)

##########################
# Run and display the results
##########################

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

wav = stream.read(waveFile)

print("Duration:",wav.duration)
print("Time Cost:",(et-st).total_seconds())
print("RTF:", (et-st).total_seconds() / wav.duration )
print("Latency:", (et-st1).total_seconds() )

#base.dynamic_display(chain.outPIPE, mapFunc=lambda packet:print(packet[packet.mainKey]))

# LINED WITH FULL MANAGEMENT OF THE HOURS OF PRIME SEEMING RECEIVED TO DAY IN THE MIDST OF LEAVES THAT EVER BORE A   BURDEN TO THE RHYMES
# NOT WHISTLED    MANAGEMENT OF THE HOURS OF PRIME SINGING RECEDED  TO DAY IN THE MIDST OF LEAVES THAT EVER BORE THE BURDEN TO THE RHYMES