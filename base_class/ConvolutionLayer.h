#ifndef _CONVOLUTION_H_
#define _CONVOLUTION_H_

#include "Node.h"
#include<vector>
#include<iostream>
class ConvolutionLayer{
public:
  int conkernel_row, conkernel_col, step, input_row, input_col;
  std::vector<std::vector<double>>* input;
  std::vector<std::vector<double>>* output;
  std::vector<std::vector<double>>* conkernel;
  ConvolutionLayer(int r, int c, int s);
  void forward();
  void backward();
  void load(std::vector<std::vector<double>>* i){input = i; input_row = i->size(); input_col = (*i)[0].size();} //load data from outside
  double g(int row, int col); //get data from input
  double g2(int row, int col); //get data from kernel
  void w(int row, int col, double data); //write data to output
};

#endif
