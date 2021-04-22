# ExKaldi-RT: An Online Speech Recognition Extension Toolkit of Kaldi 
![exkaldi-rt](https://github.com/wangyu09/exkaldi-rt/workflows/exkaldi-rt/badge.svg)

ExKaldi-RT is an online ASR toolkit for Python language.
It based on Kaldi's _LatticeFasterDecoder_.

ExKaldi-RT has these features:

1. Easy to build an online ASR pipeline with Python.

2. Use DNN acoustic model trained with DL framesworks, such as TensorFlow and PyTorch.

3. Easy to custimize original functions for, such as voice activity detection (VAD) and denoising. 

4. Support network transmission.

We tested our toolkit using Kaldi version 5.5, commit acff3f65640715f22252f143df7c3e1997899163 .

# Branch V1.1 
This is a backup of previous version which we had done our experiments.
We will upload the experimental data in this branch.

## Installation

If you plan to use ExKaldi-RT on the server, 
please make sure that Kaldi has been installed and compiled successfully before installing ExKaldi-RT.
Then follow the steps below to install ExKaldi-RT package.

1. Clone ExKaldi-RT repository.
```shell
git clone https://github.com/wangyu09/exkaldi-rt.git
``` 

2. Copy these directories into Kaldi source folder.
```shell
cp -r exkaldi-rt/exkaldirtc $KALDI_ROOT/src/
cp -r exkaldi-rt/exkaldirtcbin $KALDI_ROOT/src/
```

3. Go to source directories and compile C++ source programs.
```shell
export EXKALDIRTROOT=`pwd`

cd $KALDI_ROOT/src/exkaldirtc
make depend
make
cd $KALDI_ROOT/src/exkaldirtcbin
make depend
make
```

4. Go back to exkaldi-rt derectory and install ExKaldi-RT Python package.
```shell
cd $EXKALDIRTROOT
sudo apt-get install libjack-jackd2-dev portaudio19-dev libportaudio2
bash quick_install.sh
```

5. Check.
```shell
python -c "import exkaldirt"
```
