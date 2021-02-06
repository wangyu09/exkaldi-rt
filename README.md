# ExKaldi2: An Online Speech Recognition Toolkit for Python
![exkaldi2](https://github.com/wangyu09/exkaldi-online/workflows/exkaldionline/badge.svg)

This kaldi using acff3f65640715f22252f143df7c3e1997899163 commit.

## Installation

Before installing ExKaldi2, please ensure that Kaldi has been installed and compiled successfully.

1. Clone ExKaldi2 repository.
```git clone https://github.com/wangyu09/exkaldi-online.git``` 

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
```
sudo apt-get install libportaudio2
bash quick_install.sh
```

5. Check.
```
python -c "import exkaldi2"
```

