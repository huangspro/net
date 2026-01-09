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
  std::vector<std::vector<Var*>> layer_output;
  std::vector<std::vector<Var*>> conkernel;
  ConvolutionLayer(int r, int c, int input_r, int input_c, int s);
  
  //computing function
  void forward();
  void backward();
  
  //these function dodn't check the input or output size of the next or previous layers, if the number of neutro doesn't match, the program will crash
  void connect_to_next_conlayer_input(std::vector<std::vector<Var*>>); //connect to the next convolutional layer
  void connect_to_last_conlayer_output(std::vector<std::vector<Var*>>); //connect to the last convolutional layer
  void connect_to_next_layer_input(std::vector<std::vector<Var*>>); //connect to the next common layer
  void connect_to_last_layer_output(std::vector<std::vector<Var*>>); //connect to the last common layer
  void load_data_from_outside(std::vector<std::vector<double>>&); //load input data from outside
  
  //tool function
  double g(int row, int col); //get data from input
  Var* g2(int row, int col); //get data from kernel
  void w(int row, int col, double data); //write data to output
  double cal(int, int);//calculate out the output of input
};

#endif
