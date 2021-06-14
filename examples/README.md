There are some examples based on mini_librispeech corpus.
You can run the Kaldi's standard recipe to get the `final.mdl`, `HCLG.fst`, `words.txt` files and 
train an acoustic model with tensorflow. We give an example script `train_DNN_mfcc.py` to train such an acoustic model. 
We also prepared some pre-trained resources including: 
```shell
model.h5 # a tensorflow weights model
final.mdl # kaldi acoustic model (including HMM)
HCLG.fst # WFST decoding graph
words.txt # words to ids lexicon
```
You can download them from google drive.
```shell
https://drive.google.com/drive/folders/1AwCivR3QKVpWTuHYpOY9Wvvfntj8xehl?usp=sharing
```

After prepared these files, you can run the following scripts.

1. `01_read_decode.py`

This is a script to read data stream from file and do online decoding.

2. `01_record_decode.py`

This is a script to record data stream from microphone and do online decoding.