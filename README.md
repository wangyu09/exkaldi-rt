# ExKaldi-RT: An Online Speech Recognition Extension Toolkit of Kaldi 
![exkaldi-rt](https://github.com/wangyu09/exkaldi-rt/workflows/exkaldi-rt/badge.svg)

# Version 1.2

In branch V1.2, we are improving ExKaldi-RT from the following aspects:

1. Instead of _subprocess_ in Python, use Pybind to build the interface with C++ library.

2. Instead of mutiple threads, use mutiple processes to drive each components.

3. Improve the Packet to carry more infomation not only the data.

4. It is able to connect components parallelly to perform multiple tasks (under designing).

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

3. Install Pybind11.
```shell
pip3 install pybind11
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
make pybind
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
