# ExKaldi-RT: An Online Speech Recognition Extension Toolkit of Kaldi 
![exkaldi-rt](https://github.com/wangyu09/exkaldi-rt/workflows/exkaldi-rt/badge.svg)

ExKaldi-RT is an online ASR toolkit for Python language.
It reads realtime streaming audio and do online feature extraction, probability computation, and online decoding. It based on Kaldi's _LatticeFasterDecoder_.

ExKaldi-RT has these features:

1. Easy to build an online ASR pipeline with Python with low latency.

2. Use DNN acoustic model trained with DL framesworks, such as TensorFlow and PyTorch.

3. Easy to custimize original functions for, such as voice activity detection (VAD) and denoising. 

4. Support network transmission.

We tested our toolkit using Kaldi version 5.5, commit acff3f65640715f22252f143df7c3e1997899163 .

# Version 1.2

1. Instead of _subprocess_ in Python, use Pybind to build the interface with C++ library.
(It is being gradually completed.)

2. Still use multiple threads to drive each components ( We have tried to use multiprocessing, but we have encountered some difficulties in data communication between different processes, and are considering solutions. Multi-threading may cause a certain amount of resource congestion and affect the realtime factor, so we still plan to use multi-process in future versions.)

3. Improve the Packet to carry more infomation.

4. It is able to connect components parallelly to perform multiple tasks.

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
cd exkaldi-rt
cp -r exkaldirtc $KALDI_ROOT/src/
cp -r exkaldirtcbin $KALDI_ROOT/src/
```

3. Install Pybind11.
```shell
pip3 install pybind11
```

4. Go to source directories and compile C++ source programs.
If you have installed Pybind11, please ignore the compile error: "fatal error: pybind11/pybind11.h: No such file or directory"
```shell
export EXKALDIRTROOT=`pwd`

cd $KALDI_ROOT/src/exkaldirtc
make depend
make
cd $KALDI_ROOT/src/exkaldirtcbin
make -i depend 
make
make pybind
```

5. Go back to exkaldi-rt derectory and install ExKaldi-RT Python package.
```shell
cd $EXKALDIRTROOT
sudo apt-get install libjack-jackd2-dev portaudio19-dev libportaudio2
bash quick_install.sh
```

6. Check.
```shell
python -c "import exkaldirt"
```
