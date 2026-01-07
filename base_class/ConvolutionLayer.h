#ifndef _CONVOLUTION_H_
#define _CONVOLUTION_H_

#include "Node.h"

class ConvolutionLayer{
  int conkernel_row, conkernel_col;
  bool fill; //whether to fill the boundary to keep the size of input 
  vector<vector<double>> input;
  vector<vector<double>> output;
  ConvolutionLayer();
  void forward();
  void backward();
};

#endif
