# ExKaldi2: An Online Speech Recognition Toolkit for Python
![exkaldi2](https://github.com/wangyu09/exkaldi2/workflows/exkaldi2/badge.svg)

ExKaldi2 is an online ASR toolkit for Python language.
It based on Kaldi lattice faster decoder.

ExKaldi2 has these features:

1. We provided a basic vad function with Google webrtc, 
but you can apply your own VAD function, such as DNN
vad function trained with DL frameworks.

2. It supported extracting Spectrogram, fBank, MFCC acoustic feature in current version. Basides, you can design your own feature function.

3. ExKaldi2 uses DNN acoustic model trained with DL framesworks.
In current version, we only support WFST online decoder. 

4. It's easy to use RNN LM trained by DL frameworks to rescore N-Best result.

5. Support network transmition.

This kaldi using acff3f65640715f22252f143df7c3e1997899163 commit.

## Installation

Before installing ExKaldi2, please ensure that Kaldi has been installed and compiled successfully.

1. Clone ExKaldi2 repository.
```shell
git clone https://github.com/wangyu09/exkaldi2.git
``` 

2. Copy these directories into Kaldi source folder.
```shell
mv exkaldi-online/exkaldionline $KALDI_ROOT/src/
mv exkaldi-online/exkaldionlinebin $KALDI_ROOT/src/
```

3. Compile the source programs.
```shell
cd $KALDI_ROOT/src/exkaldionline
make depend
make
cd $KALDI_ROOT/src/exkaldionlinebin
make depend
make
```

4. Go back to exkaldi-online derectory and install ExKaldi2 python package.
```shell
sudo apt-get install libportaudio2
bash quick_install.sh
```

5. Check.
```shell
python -c "import exkaldi2"
```

