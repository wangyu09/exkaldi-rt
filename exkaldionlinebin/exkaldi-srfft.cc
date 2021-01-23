#include<iostream>
#include<vector>

#include "matrix/srfft.h"
#include "matrix/kaldi-vector.h"

int main(int argc, char *argv[])
{ 
  using namespace kaldi;

  int padded_window_size = atoi(argv[2]);

  int num_frames, points;
  std::cin >> num_frames;
  std::cin >> points;
  std::cin.ignore();

  Vector<BaseFloat> mywave;
  mywave.Resize(padded_window_size);
  mywave.SetZero();

  SplitRadixRealFft<BaseFloat> *srfft2 = new SplitRadixRealFft<BaseFloat>(padded_window_size);

  for (int i=0; i<num_frames; i++)
  {
    mywave.SetZero();

    for (int j=0; j<points; j++)
    {
      std::cin >> mywave(j);
    }
    
    srfft2->Compute(mywave.Data(), true);

    mywave(0) = ( mywave(0) + mywave(1) ) / 2.0;
    mywave(1) = mywave(0) - mywave(1);

    for(int k=0; k<padded_window_size; k++)
    {
      std::cout << mywave(k) << " ";
    }
  }

  delete srfft2;

  std::cout << std::endl;
  std::cout.flush();

  return 0;
}