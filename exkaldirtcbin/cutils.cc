#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<iostream>
#include<limits>

#include"base/kaldi-math.h"
#include"matrix/kaldi-matrix.h"
#include"matrix/srfft.h"
#include"matrix/kaldi-vector.h"
#include"feat/feature-functions.h"

using namespace kaldi;

float get_float_floor(){
  return std::numeric_limits<float>::epsilon();
}

pybind11::array_t<double> dither(pybind11::array_t<double> frames, float factor)
{
  pybind11::buffer_info buf = frames.request();
  auto result = pybind11::array_t<double>(buf.shape);

  RandomState rstate;

  for (size_t i = 0; i < buf.shape[0]; i++){
    for (size_t j = 0; j < buf.shape[1]; j++){
      *result.mutable_data(i,j) = (*frames.data(i,j)+ RandGauss(&rstate) * factor);
    }
  }

  return result;
}

pybind11::array_t<double> srfft(pybind11::array_t<double> frames, size_t padded_window_size)
{ 
  size_t dim0 = frames.shape(0);
  size_t dim1 = frames.shape(1);
  size_t dim2 = padded_window_size/2;
  size_t dim3 = 2;
  std::vector<size_t> shape{dim0,dim2,dim3};
  auto result = pybind11::array_t<double>(shape);

  SplitRadixRealFft<BaseFloat> *srfft2 = new SplitRadixRealFft<BaseFloat>(padded_window_size);
  Vector<BaseFloat> mywave;
  mywave.Resize(padded_window_size);

  for (size_t i = 0; i < dim0; i++){
    mywave.SetZero();
    for (size_t j = 0; j < dim1; j++){
      mywave(j) = *frames.data(i,j);
    }
    srfft2->Compute(mywave.Data(), true);
    mywave(0) = ( mywave(0) + mywave(1) ) / 2.0;
    mywave(1) = mywave(0) - mywave(1);
    for (size_t k = 0; k < dim2; k++){
      *result.mutable_data(i,k,0) = mywave(2*k);
      *result.mutable_data(i,k,1) = mywave(2*k+1);
    }
  }
  return result;
}

pybind11::array_t<double> splice_feat(pybind11::array_t<double> frames, size_t left, size_t right)
{
  size_t num_frame = frames.shape(0);
  size_t feat_dim = frames.shape(1);

  Matrix<BaseFloat> feats;
  feats.Resize(num_frame,feat_dim);
  feats.SetZero();

  for (size_t i=0; i<num_frame; i++){
    for (size_t j=0; j<feat_dim; j++){
      feats(i,j) = *frames.data(i,j);
    }
  }

  Matrix<BaseFloat> new_feats;
  SpliceFrames(feats, left, right, &new_feats);

  size_t new_feat_dim = new_feats.NumCols();
  std::vector<size_t> shape{num_frame,new_feat_dim};
  auto result = pybind11::array_t<double>(shape);

  for (int m=0; m<num_frame; m++){
    for (int n=0; n<new_feat_dim; n++){
      *result.mutable_data(m,n) = new_feats(m,n);
    }
  }

  return result;
}

pybind11::array_t<double> add_deltas(pybind11::array_t<double> frames, size_t order, size_t window)
{
  size_t num_frame = frames.shape(0);
  size_t feat_dim = frames.shape(1);

  Matrix<BaseFloat> feats;
  feats.Resize(num_frame,feat_dim);
  feats.SetZero();

  for (size_t i=0; i<num_frame; i++){
    for (size_t j=0; j<feat_dim; j++){
      feats(i,j) = *frames.data(i,j);
    }
  }

  DeltaFeaturesOptions dopts(order,window);
  Matrix<BaseFloat> new_feats;
  ComputeDeltas(dopts, feats, &new_feats);

  size_t new_feat_dim = new_feats.NumCols();
  std::vector<size_t> shape{num_frame,new_feat_dim};
  auto result = pybind11::array_t<double>(shape);

  for (int m=0; m<num_frame; m++){
    for (int n=0; n<new_feat_dim; n++){
      *result.mutable_data(m,n) = new_feats(m,n);
    }
  }

  return result;
}

PYBIND11_MODULE(cutils,m){
  m.doc() = "ExKaldi-RT c++ utils";
  m.def("get_float_floor",&get_float_floor,"Get float floor value.");
  m.def("dither",&dither,"Dither.");
  m.def("srfft",&srfft,"Do split radix real FFT.");
  m.def("splice_feat",&splice_feat,"Splice feature.");
  m.def("add_deltas",&add_delta,"Add delta features.");
}