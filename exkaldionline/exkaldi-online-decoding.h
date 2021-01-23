#ifndef EXKALDI_ONLINE_DECODING_H_
#define EXKALDI_ONLINE_DECODING_H_

#include <string>
#include <vector>
#include <ctime>

#include"base/kaldi-common.h"
#include"itf/options-itf.h"
#include"itf/decodable-itf.h"
#include"hmm/transition-model.h"
#include"matrix/kaldi-matrix.h"
#include"decoder/lattice-faster-decoder.h"
#include"fst/types.h"
#include"lat/lattice-functions.h"
#include"lat/word-align-lattice.h"
#include"lat/determinize-lattice-pruned.h"
#include"online2/online-endpoint.h"

namespace exkaldi {

using namespace kaldi;

const int CLOCKS_PER_MSEC = CLOCKS_PER_SEC / 1000;

void TimeDelay(int msec);

struct ExkaldiDecodableOnlineOptions {

  BaseFloat acoustic_scale;
  int32 chunk_frames;

  ExkaldiDecodableOnlineOptions():
            acoustic_scale(0.1),
            chunk_frames(64){}
    
  void Register(OptionsItf *opts) {
    opts->Register("acoustic-scale", &acoustic_scale,
                   "Scaling factor for acoustic likelihoods");
    opts->Register("chunk-frames", &chunk_frames,
                   "Number of frames for each decoding.");
  }
};

class ExkaldiDecodableOnline: public DecodableInterface 
{
  public:
    ExkaldiDecodableOnline(TransitionModel &trans_model,
                           const ExkaldiDecodableOnlineOptions &opts);
    // Get scaled log likelihood of acoustic model
    BaseFloat LogLikelihood(int32 frame, int32 index);
    bool IsLastFrame(int32 frame) const;
    int32 NumFramesReady() const; 
    int32 NumIndices() const { return trans_model_.NumTransitionIds(); }
    bool RecieveFrames(const int timeout, const int timescale);
    bool ArrivedLastChunk() { return last_frame_id_ != -1; }

    void Terminate(){ termination_ = true; }
    bool IsTermination() { return termination_ == true; }

  private:
    const ExkaldiDecodableOnlineOptions opts_;
    TransitionModel &trans_model_;
    int32 begin_frame_;
    int32 last_frame_id_;
    int32 frames_ready_;
    int32 available_frames_;

    Matrix<BaseFloat> loglikes_;
    int32 pdf_ids_;
    bool termination_ = false;
};

struct ExkaldiOnlineDecodingConfig {
  
  LatticeFasterDecoderConfig decoder_opts;
  ExkaldiDecodableOnlineOptions decodable_opts;
  
  ExkaldiOnlineDecodingConfig() {  decodable_opts.acoustic_scale = 0.1; }

  ExkaldiOnlineDecodingConfig(LatticeFasterDecoderConfig &dropts,
                              ExkaldiDecodableOnlineOptions &deopts):
                              decoder_opts(dropts),
                              decodable_opts(deopts){ }
    
  void Register(OptionsItf *opts) {
    decoder_opts.Register(opts);
    decodable_opts.Register(opts);
  }
};

class ExkaldiSingalUtteranceDecoder
{
  public:
    ExkaldiSingalUtteranceDecoder(const ExkaldiOnlineDecodingConfig &config,
                                  TransitionModel &tmodel,
                                  fst::Fst<fst::StdArc> &fst);
    
    void AdvanceDecoding();
    void FinalizeDecoding();
    int32 NumFramesDecoded() const;
    void GetLattice(bool end_of_utterance,
                  CompactLattice *clat) const;
    void GetBestPath(bool end_of_utterance,
                   Lattice *best_path) const;
    bool EndpointDetected(const OnlineEndpointConfig &config, 
                          BaseFloat frame_shift_in_seconds);

    const LatticeFasterOnlineDecoder &Decoder() const { return decoder_; }

    ~ExkaldiSingalUtteranceDecoder() { }

    bool RecieveFrames(const int timeout, const int timescale) { return decodable_.RecieveFrames(timeout,timescale); }

    bool IsLastDecoding(){ return decodable_.ArrivedLastChunk(); }

    int32 IsTermination(){ return decodable_.IsTermination(); }

  private:
    ExkaldiOnlineDecodingConfig config_;
    TransitionModel &tmodel_;
    ExkaldiDecodableOnline decodable_;
    LatticeFasterOnlineDecoder decoder_;
};

int EmitPartialResult(const Lattice &lat);

int EmitFinalResult(CompactLattice &clat, 
                    BaseFloat acwt, 
                    BaseFloat lmwt, 
                    WordBoundaryInfo *wbi,
                    TransitionModel &trans_model,
                    int32 N_best);
  //Get the n-best

void WaitForOver(int32 timeout, int32 timescale);
}

#endif 