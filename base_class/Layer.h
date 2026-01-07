/*
This file contains some basic neural net layers
Input Layer, Softmax Layer, Hidden Layer, Nonlinear Layer, MeanSquareErrorLayer
1. an input layer contains input nodes and multiply inputs with weights
2. a Softmax layer contains input nodes and calculate the softmax function output of inputs
3. a hidden layer can receive the inputs of the last layer and calculate the output with weights
4. a nonlinear layer can transform the input data with a nonlinear function like relu or tanh and output
5. a loss layer can implement the loss function like mean square or cross-entropy
noted that an output layer is a combination of hidden layer and other layer
*/

#ifndef _LAYER_H_
#define _LAYER_H_

#define learning_ratio - 0.001
#include "Node.h"

#include <vector>
#include <iostream>

// This layer is the core layer of a neural net, containing a nonlinear function
class NonlinearLayer {
public:
  static const int RELU = 1, TANH = 2, SIGMOID = 3, LINEAR = 4;

  int neuron, function_type;
  std::vector<Var *> input, layer_output;
  std::vector<Ope *> opes;

  NonlinearLayer(int n, int f);
  void forward();
  void backward();
  void connect_to_last_layer_output(std::vector<Var *> last_layer_output);
  void connect_to_next_layer_input(std::vector<Var *> &next_layer_input);
  ~NonlinearLayer();
};

// This layer is an input layer, which can tackle data input
class InputLayer {
public:
  int neuron;
  std::vector<Var *> input, weight, bias, mul_output, add_output, layer_output;
  std::vector<Ope *> mul, add;

  InputLayer(int n);
  void forward();
  void backward();
  void train();
  void input_data(std::vector<double> one_data);
  ~InputLayer();
};

// This layer is a hiddenlayer, containing two dimensional vector to store the weights for each unit
class HiddenLayer {
public:
  int neuron, last_layer_neuron_number;
  std::vector<Var *> input, bias, layer_output;
  std::vector<std::vector<Var *>> weights, mul_output;
  std::vector<Ope *> superadd;
  std::vector<std::vector<Ope *>> mul;

  HiddenLayer(int n, int last_layer);
  void forward();
  void backward();
  void train();
  ~HiddenLayer();
};

// This layer is for calculate the softmax function output
class SoftmaxLayer {
public:
  int neuron;
  std::vector<Var *> input, e_output, layer_output;
  Var *superadd_output, *dev_output;
  std::vector<Ope *> exp, mul;
  Ope *superadd, *dev;

  SoftmaxLayer(int n);
  void forward();
  void backward();
  void connect_to_last_layer_output(std::vector<Var *> last_layer_output);
  void connect_to_next_layer_input(std::vector<Var *> next_layer_input);
  ~SoftmaxLayer();
};

// MeanSquareErrorLayer
class MeanSquareErrorLayer {
public:
  int neuron;
  double loss_value;
  std::vector<Var *> input, input_from_outside, minus_output, add_output, square_output, layer_output;
  std::vector<Ope *> minus, add, square;
  Ope *superadd;

  MeanSquareErrorLayer(int n);
  void forward();
  void backward();
  void load_data_from_outside(std::vector<double> one_data);
  ~MeanSquareErrorLayer();
};

// CrossEntropyLossLyaer
class CrossEntropyLossLayer {
public:
  double loss_value;
  int neuron;
  std::vector<Var *> input, input_from_outside, ln_output, mul_output, superadd_output, layer_output;
  std::vector<Ope *> ln, mul;
  Ope *superadd, *minus;

  CrossEntropyLossLayer(int n);
  void forward();
  void backward();
  void load_data_from_outside(std::vector<double> one_data);
  ~CrossEntropyLossLayer();
};

#endif
