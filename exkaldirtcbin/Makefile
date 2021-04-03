
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = exkaldi-add-deltas exkaldi-dither exkaldi-online-decoder  \
           exkaldi-read-wave exkaldi-splice-feats exkaldi-srfft get-float-floor

OBJFILES = 

TESTFILES =

ADDLIBS = ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../decoder/kaldi-decoder.a \
          ../base/kaldi-base.a ../feat/kaldi-feat.a ../hmm/kaldi-hmm.a \
          ../lat/kaldi-lat.a ../exkaldirtc/exkaldi-online.a ../fstext/kaldi-fstext.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a \
          ../online2/kaldi-online2.a

include ../makefiles/default_rules.mk
