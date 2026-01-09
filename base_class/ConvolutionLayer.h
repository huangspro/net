#ifndef _CONVOLUTION_H_
#define _CONVOLUTION_H_

#include "Node.h"
#include<vector>
#include<iostream>
#include<random>

//this layer is defined alone, for its speciality. 0-based index, defaults not to fill
//if the inputsize, kernelsize and step don't be chosen propely, errors would occur and exit the program
class ConvolutionLayer{
public:
  int conkernel_row, conkernel_col, step, input_row, input_col;
  std::vector<std::vector<Var*>> input;
  std::vector<std::vector<Var*>> output;
  std::vector<std::vector<Var*>> conkernel;
  ConvolutionLayer(int r, int c, int input_r, int input_c, int s);
  void forward();
  void backward();
  void load(std::vector<std::vector<double>>&); //load input data from outside
  double g(int row, int col); //get data from input
  Var* g2(int row, int col); //get data from kernel
  void w(int row, int col, double data); //write data to output
  double cal(int, int);//calculate out the output of input
};

#endif
