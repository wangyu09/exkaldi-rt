# ExKaldi-RT: An Online Speech Recognition Extension Toolkit based on Kaldi 
![exkaldi-rt](https://github.com/wangyu09/exkaldi-rt/workflows/exkaldi-rt/badge.svg)

ExKaldi-RT is an online ASR toolkit for Python language.
It based on Kaldi lattice faster decoder.

ExKaldi-RT has these features:

1. We provided a basic voice activity detection (VAD) function based on Google Webrtc VAD, 
but you can apply your own VAD function, including a DNN model trained with the deep learning frameworks.

2. It supported extracting Spectrogram, fBank, MFCC, LDA+MLLT (and their mixture) acoustic feature in current version. 
Besides, you can design your original feature function.

3. ExKaldi-RT uses DNN acoustic model trained with DL framesworks.
In current version, we only support online decoder based on WFST decoding graph. 

4. It's easy to use RNN langauge model trained with DL frameworks to rescore N-Best results.

5. Support network transmission and customizing compression algorithm.

We test our toolkit using Kaldi version 5.5, commit acff3f65640715f22252f143df7c3e1997899163 .

## Installation

If you plan to use ExKaldi-RT on the server, 
please make sure that before installing ExKaldi-RT, Kaldi has been installed and compiled successfully.

1. Clone ExKaldi-RT repository.
```shell
git clone https://github.com/wangyu09/exkaldi-rt.git
``` 

2. Copy (or move) these directories into Kaldi source folder.
```shell
mv exkaldi-rt/exkaldionline $KALDI_ROOT/src/
mv exkaldi-rt/exkaldionlinebin $KALDI_ROOT/src/
```

3. Go to source directories and compile C++ source programs.
```shell
export EXKALDIRTROOT=`pwd`

cd $KALDI_ROOT/src/exkaldionline
make depend
make
cd $KALDI_ROOT/src/exkaldionlinebin
make depend
make
```

4. Go back to exkaldi-rt derectory and install ExKaldi-RT python package.
```shell
cd $EXKALDIRTROOT
sudo apt-get install libportaudio2
bash quick_install.sh
```

5. Check.
```shell
python -c "import exkaldi_rt"
```
