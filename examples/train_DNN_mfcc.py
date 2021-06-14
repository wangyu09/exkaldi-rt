import os
import random
import datetime

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import exkaldi
from exkaldi import args

from neural_networks import make_DNN_acoustic_model

exkaldi.info.set_timeout(1800)

#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[-1], True)
#tf.config.experimental.set_visible_devices(gpus[-1], 'GPU')

KALDI_ROOT = exkaldi.info.KALDI_ROOT
rootdir = f"{KALDI_ROOT}/egs/mini_librispeech/s5"

args.add("--testModelDir", abbr="-t", dtype=str, default=, description="If not empty, do test step.")
args.add("--root", abbr="-r", dtype=str, default=rootdir, description="The root dir of data.")
args.add("--feat", abbr="-f", dtype=str, default="mfcc", description="Feature.")
args.add("--cmn", abbr="-c", dtype=bool, default=True, description="CMVN.")
args.add("--delta", abbr="-d", dtype=int, default=2, description="Delta feature.")
args.add("--splice", abbr="-s", dtype=int, default=10, description="Splice feature.")
args.add("--batchSize", abbr="-b", dtype=int, default=128, description="Mini batch size.")
args.add("--epoch", abbr="-e", dtype=int, default=30, description="Epoches.")
args.parse()

def prepare_data(training=True):
  
  if training:
    flag = "train_clean_5"
  else:
    flag = "dev_clean_2"
  
  print(f"{flag}: Load feature...")
  featsFile = f"{args.root}/{args.feat}/raw_{args.feat}_{flag}.*.ark"
  feats = exkaldi.load_feat(featsFile)

  if args.cmn:
    print(f"{flag}: Use cmvn...")
    cmvnFile = f"{args.root}/{args.feat}/cmvn_{flag}.ark"
    cmvn = exkaldi.load_cmvn(cmvnFile)
    feats = exkaldi.use_cmvn(feats,cmvn,utt2spk=f"{args.root}/data/{flag}/utt2spk")
    del cmvn

  if args.delta > 0:
    print(f"{flag}: Add delta...")
    feats = feats.add_delta(args.delta)

  if args.splice > 0:
    print(f"{flag}: Splice feature...")
    feats = feats.splice(args.splice)
  
  feats = feats.to_numpy()
  featDim = feats.dim
  
  print(f"{flag}: Load alignment...")
  ali = exkaldi.load_ali( f"{args.root}/exp/tri3b_ali_{flag}/ali.*.gz" )
  print(f"{flag}: Get pdf alignment...")
  pdfAli = ali.to_numpy(aliType="pdfID",hmm=f"{args.root}/exp/tri3b_ali_{flag}/final.mdl")
  del ali

  feats.rename("feat")
  pdfAli.rename("pdfID")
  print(f"{flag}: Tuple dataset...")
  dataset = exkaldi.tuple_dataset([feats, pdfAli], frameLevel=True)
  random.shuffle(dataset)

  return featDim, dataset

def make_generator(dataset):
  dataIndex = 0
  datasetSize = len(dataset)
  while True:
    if dataIndex >= datasetSize:
      random.shuffle(dataset)
      dataIndex = 0
    one = dataset[dataIndex]
    dataIndex += 1
    yield one.feat[0], one.pdfID[0]

class ModelSaver(keras.callbacks.Callback):

  def __init__(self, model, outDir):
    self.model = model
    self.outDir = outDir
  
  def on_epoch_end(self, epoch, logs={}):  
    outFile = os.path.join(self.outDir,f"model_ep{epoch+1}.h5")
    self.model.save_weights(outFile)

def train():

  # ------------- Prepare data for dnn training ----------------------
  stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  outDir = f"dnn_exp/out_{stamp}"
  exkaldi.utils.make_dependent_dirs(outDir, pathIsFile=False)
  args.save( os.path.join(outDir,"conf") )

  #------------------------ Training and Validation dataset-----------------------------
  hmm = exkaldi.load_hmm(f"{args.root}/exp/tri3b_ali_train_clean_5/final.mdl")
  pdfDim = hmm.info.pdfs
  del hmm

  print('Prepare Data Iterator...')
  # Prepare fMLLR feature files
  featDim, trainDataset = prepare_data()
  traindataLen = len(trainDataset)

  train_gen = tf.data.Dataset.from_generator(
                                      lambda: make_generator(trainDataset),
                                      (tf.float32, tf.int32),
                                      (tf.TensorShape([featDim,]), tf.TensorShape([])),
                              ).batch(args.batchSize).prefetch(3)
  steps_per_epoch = traindataLen//args.batchSize

  featDim, devDataset = prepare_data(training=False)
  devdataLen = len(devDataset)
  dev_gen = tf.data.Dataset.from_generator(
                                      lambda: make_generator(devDataset),
                                      (tf.float32, tf.int32),
                                      (tf.TensorShape([featDim,]), tf.TensorShape([])),
                              ).batch(args.batchSize).prefetch(3)
  validation_steps = devdataLen//args.batchSize

  #------------------------ Train Step -----------------------------
  model = make_DNN_acoustic_model(featDim,pdfDim)
  #model.summary()

  model.compile(
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics = keras.metrics.SparseCategoricalAccuracy(),
            optimizer = keras.optimizers.SGD(0.08,momentum=0.0),
        )

  def lrScheduler(epoch):
    if epoch > 25:
        return 0.001
    elif epoch > 22:
        return 0.0025
    elif epoch > 19:
        return 0.005
    elif epoch > 17:
        return 0.01
    elif epoch > 15:
        return 0.02
    elif epoch > 10:
        return 0.04
    else:
        return 0.08

  model.fit(
          x = train_gen,
          steps_per_epoch=steps_per_epoch,
          epochs=args.epoch,

          validation_data=dev_gen,
          validation_steps=validation_steps,
          verbose=1,

          initial_epoch=0,
          callbacks=[
                      keras.callbacks.EarlyStopping(patience=5, verbose=1),
                      keras.callbacks.TensorBoard(log_dir=outDir),
                      keras.callbacks.LearningRateScheduler(lrScheduler),
                      ModelSaver(model,outDir),         
                  ],
              )

def compute_dev_wer():

  flag = "dev_clean_2"
  
  featsFile = f"{args.root}/{args.feat}/raw_{args.feat}_{flag}.*.ark"
  feats = exkaldi.load_feat(featsFile)

  if args.cmn:
    print("Use cmvn...")
    cmvnFile = f"{args.root}/{args.feat}/cmvn_{flag}.ark"
    cmvn = exkaldi.load_cmvn(cmvnFile)
    feats = exkaldi.use_cmvn(feats,cmvn,utt2spk=f"{args.root}/data/{flag}/utt2spk")
    del cmvn

  if args.delta > 0:
    print("Add delta...")
    feats = feats.add_delta(args.delta)

  if args.splice > 0:
    print("Splice feature...")
    feats = feats.splice(args.splice)
  
  feats = feats.to_numpy()
  featDim = feats.dim

  hmm = exkaldi.load_hmm(f"{args.root}/exp/tri3b_ali_train_clean_5/final.mdl")
  pdfDim = hmm.info.pdfs
  phoneDim = hmm.info.phones
  del hmm
  
  print("featDim:",featDim,"pdfDim:",pdfDim,"phoneDim:",phoneDim)
  minWER = None

  try:
    for ModelPathID in range(args.epoch,0,-1):
      #ModelPathID = args.epoch
      ModelPath = f"{args.testModelDir}/model_ep{ModelPathID}.h5"
      if not os.path.isfile(ModelPath):
        continue

      print("Use Model:",ModelPath)
      decodeOut = ModelPath[:-3]
      exkaldi.utils.make_dependent_dirs(decodeOut,pathIsFile=False)

      model = make_DNN_acoustic_model(featDim,pdfDim)
      model.load_weights(ModelPath)

      print("Forward...")
      result = {}
      for uttID in feats.keys():
        pdfP = model(feats[uttID],training=False)
        result[uttID] = exkaldi.nn.log_softmax(pdfP.numpy(),axis=1)

      amp = exkaldi.load_prob(result)
      hmmFile = f"{args.root}/exp/tri3b_ali_dev_clean_2/final.mdl"
      HCLGFile = f"{args.root}/exp/tri3b/graph_tgsmall/HCLG.fst"
      table = f"{args.root}/exp/tri3b/graph_tgsmall/words.txt"
      trans = f"{args.root}/data/dev_clean_2/text"

      print("Decoding...")
      lat = exkaldi.decode.wfst.nn_decode(
                                          prob=amp.subset(chunks=4), 
                                          hmm=hmmFile, 
                                          HCLGFile=HCLGFile, 
                                          symbolTable=table,
                                          beam=10,
                                          latBeam=8,
                                          acwt=0.1,
                                          minActive=200,
                                          maxActive=7000,
                                          outFile=os.path.join(decodeOut,"lat")
                                        )
      lat = exkaldi.merge_archives(lat)

      print("Scoring...")
      for LMWT in range(1,10,1):
        #newLat = lat.add_penalty(penalty)
        result = lat.get_1best(table,hmmFile,lmwt=LMWT,acwt=0.1,phoneLevel=False)
        result = exkaldi.hmm.transcription_from_int(result,table)
        result.save( os.path.join(decodeOut,f"trans.{LMWT}") )

        score = exkaldi.decode.score.wer(ref=trans,hyp=result,mode="present")
        print("LMWT: ",LMWT ,"WER: ",score.WER)
        if minWER == None or score.WER < minWER[0]:
          minWER = (score.WER, LMWT, ModelPath)
  finally:
    if minWER is not None:
      werOut = os.path.basename(decodeOut)
      print("Best WER:",minWER)
      with open(f"{args.testModelDir}/best_wer","w") as fw:
        fw.write(str(minWER))

if __name__ == "__main__":
  if args.testModelDir == "":
    train()
  else:
    compute_dev_wer()
  
