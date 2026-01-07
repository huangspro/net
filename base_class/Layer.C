#include "Layer.h"

NonlinearLayer::NonlinearLayer(int n, int f): neuron(n), function_type(f) {
  for (int i = 0; i < n; i++) {
    //create all nodes
    input.push_back(new Var(0, 0));
    layer_output.push_back(new Var(0, 0));
    switch (f) { //judge nonlinear function type
    case RELU:
      opes.push_back(new Relu());
      break;
    case TANH:
      opes.push_back(new Tanh());
      break;
    case SIGMOID:
      opes.push_back(new Sigmoid());
      break;
    case LINEAR:
      opes.push_back(new Nothing());
      break;
    }
    //load all operation nodes
    opes[i] -> load(input[i], layer_output[i]);
  }
}

void NonlinearLayer::forward() {
  for (auto i = opes.begin(); i != opes.end(); i++) {
    ( * i) -> forward();
  }
}

void NonlinearLayer::backward() {
  for (auto i = opes.begin(); i != opes.end(); i++) {
    ( * i) -> backward();
  }
}

void NonlinearLayer::connect_to_last_layer_output(std::vector < Var * > last_layer_output) {
  for (int i = 0; i < last_layer_output.size(); i++) {
    delete input[i];
    input[i] = last_layer_output[i];
    opes[i] -> load(input[i], layer_output[i]);
  }
}
void NonlinearLayer::connect_to_next_layer_input(std::vector < Var * > & next_layer_input) {
  for (int i = 0; i < next_layer_input.size(); i++) {
    delete layer_output[i];
    layer_output[i] = next_layer_input[i];
    opes[i] -> load(input[i], layer_output[i]);
  }
}
NonlinearLayer::~NonlinearLayer() {
  for (int i = 0; i < neuron; i++) {
    delete input[i];
    input[i] = nullptr;
    delete layer_output[i];
    layer_output[i] = nullptr;
    delete opes[i];
  }
}

//===============================================================================
InputLayer::InputLayer(int n): neuron(n) {
  for (int i = 0; i < neuron; i++) {
    //create the Nodes
    input.push_back(new Var(0, 0));
    weight.push_back(new Var(0, 0));
    bias.push_back(new Var(0, 0));
    layer_output.push_back(new Var(0, 0));
    mul_output.push_back(new Var(0, 0));
    add_output.push_back(new Var(0, 0));

    mul.push_back(new Mul());
    add.push_back(new Add());
    //load the operation nodes
    mul[i] -> load(input[i], weight[i], mul_output[i]);
    add[i] -> load(mul_output[i], bias[i], layer_output[i]);
  }
}
//forward the data
void InputLayer::forward() {
  for (auto i = mul.begin(); i != mul.end(); i++) {
    ( * i) -> forward();
  }
  for (auto i = add.begin(); i != add.end(); i++) {
    ( * i) -> forward();
  }
}
//backward the gradient
void InputLayer::backward() {
  for (auto i = add.begin(); i != add.end(); i++) {
    ( * i) -> backward();
  }
  for (auto i = mul.begin(); i != mul.end(); i++) {
    ( * i) -> backward();
  }
}
//train the layer
void InputLayer::train() {
  for (int i = 0; i < neuron; i++) {
    weight[i] -> data += learning_ratio * weight[i] -> gradient;
    bias[i] -> data += learning_ratio * bias[i] -> gradient;
  }
}
//this layer can receive data outside
void InputLayer::input_data(std::vector < double > one_data) {
  for (int i = 0; i < one_data.size(); i++) {
    input[i] -> data = one_data[i];
  }
}
InputLayer::~InputLayer() {
  for (int i = 0; i < neuron; i++) {
    delete input[i];
    input[i] = nullptr;
    delete weight[i];
    delete bias[i];
    delete mul_output[i];
    delete add_output[i];
    delete layer_output[i];
    layer_output[i] = nullptr;
    delete mul[i];
    delete add[i];
  }
}

//==============================================================

HiddenLayer::HiddenLayer(int n, int last_layer): neuron(n), last_layer_neuron_number(last_layer) {
  //layer
  //firstly, the single one dimension part
  for (int i = 0; i < n; i++) {
    bias.push_back(new Var(0, 0));
    layer_output.push_back(new Var(0, 0));
    superadd.push_back(new SuperAdd());
  }
  for (int i = 0; i < last_layer; i++) {
    input.push_back(new Var(0, 0));
  }
  //the two dimension part
  for (int i = 0; i < n; i++) {
    std::vector < Var * > tem_mul_output(last_layer);
    std::vector < Var * > tem_weights(last_layer);
    std::vector < Ope * > tem_mul(last_layer);
    for (int ii = 0; ii < last_layer; ii++) {
      tem_mul_output[ii] = new Var(0, 0);
      tem_weights[ii] = new Var(0, 0);
      tem_mul[ii] = new Mul();
      //load the multiply nodes
      tem_mul[ii] -> load(tem_weights[ii], input[ii], tem_mul_output[ii]);
    }

    mul_output.push_back(tem_mul_output);
    weights.push_back(tem_weights);
    mul.push_back(tem_mul);
  }
  //load the superadd operation nodes
  for (int i = 0; i < n; i++) {
    for (int ii = 0; ii < last_layer; ii++) {
      superadd[i] -> load_input(mul_output[i][ii]);
      superadd[i] -> load_output(layer_output[i]);
    }
  }
}
//forward the data
void HiddenLayer::forward() {
  for (int i = 0; i < neuron; i++) {
    for (int ii = 0; ii < last_layer_neuron_number; ii++) {
      mul[i][ii] -> forward();
    }
  }
  for (int i = 0; i < neuron; i++) {
    superadd[i] -> forward();
  }
}
//pass backward the gradient
void HiddenLayer::backward() {
  for (int i = 0; i < neuron; i++) {
    superadd[i] -> backward();
  }
  for (int i = 0; i < neuron; i++) {
    for (int ii = 0; ii < last_layer_neuron_number; ii++) {
      mul[i][ii] -> backward();
    }
  }
}
//train the layer
void HiddenLayer::train() {
  for (int i = 0; i < bias.size(); i++) {
    bias[i] -> data += learning_ratio * bias[i] -> gradient;
  }
  for (int i = 0; i < weights.size(); i++) {
    for (int ii = 0; ii < weights[i].size(); ii++) {
      weights[i][ii] -> data += learning_ratio * weights[i][ii] -> gradient;
    }
  }
}
//to avoid delete null pointer, input and layer_output should be nullptr after being deleted
HiddenLayer::~HiddenLayer() {
  for (int i = 0; i < last_layer_neuron_number; i++) {
    delete input[i];
    input[i] = nullptr;
    delete superadd[i];
    delete layer_output[i];
    layer_output[i] = nullptr;
    delete bias[i];
    for (int ii = 0; ii < last_layer_neuron_number; ii++) {
      delete mul[i][ii];
      delete mul_output[i][ii];
      delete weights[i][ii];
    }
  }
}

//===========================================================================
SoftmaxLayer::SoftmaxLayer(int n): neuron(n) {
  //create nodes
  superadd_output = new Var(0, 0);
  dev_output = new Var(0, 0);
  superadd = new SuperAdd();
  dev = new Dev();
  //load all the nodes
  superadd -> load_output(superadd_output);
  dev -> load(superadd_output, dev_output);
  for (int i = 0; i < n; i++) {
    input.push_back(new Var(0, 0));
    e_output.push_back(new Var(0, 0));
    layer_output.push_back(new Var(0, 0));
    exp.push_back(new Exp());
    mul.push_back(new Mul());
    //load all the nodes
    mul[i] -> load(dev_output, e_output[i], layer_output[i]);
    exp[i] -> load(input[i], e_output[i]);
    superadd -> load_input(e_output[i]);
  }
}
//pass forward data
void SoftmaxLayer::forward() {
  for (int i = 0; i < neuron; i++) {
    exp[i] -> forward();
  }
  superadd -> forward();
  dev -> forward();
  for (int i = 0; i < neuron; i++) {
    mul[i] -> forward();
  }
}
//pass the gradient backward
void SoftmaxLayer::backward() {
  for (int i = 0; i < neuron; i++) {
    mul[i] -> backward();
  }
  dev -> backward();
  superadd -> backward();
  for (int i = 0; i < neuron; i++) {
    exp[i] -> backward();
  }
}
//a Softmax layer can should have the ability to connect with the last or the next layer
void SoftmaxLayer::connect_to_last_layer_output(std::vector < Var * > last_layer_output) {
  for (int i = 0; i < last_layer_output.size(); i++) {
    delete input[i];
    input[i] = last_layer_output[i];
  }
}
void SoftmaxLayer::connect_to_next_layer_input(std::vector < Var * > next_layer_input) {
  for (int i = 0; i < next_layer_input.size(); i++) {
    delete next_layer_input[i];
    next_layer_input[i] = layer_output[i];
  }
}
SoftmaxLayer::~SoftmaxLayer() {
  delete superadd_output;
  delete dev_output;
  delete superadd;
  delete dev;
  for (int i = 0; i < neuron; i++) {
    delete mul[i];
    delete input[i];
    input[i] = nullptr;
    delete e_output[i];
    delete layer_output[i];
    layer_output[i] = nullptr;
  }
}

//====================================================================
MeanSquareErrorLayer::MeanSquareErrorLayer(int n): neuron(n) {
  superadd = new SuperAdd();
  layer_output.push_back(new Var(0, 0));
  for (int i = 0; i < n; i++) {
    //create nodes
    input.push_back(new Var(0, 0));
    input_from_outside.push_back(new Var(0, 0));
    minus_output.push_back(new Var(0, 0));
    add_output.push_back(new Var(0, 0));
    square_output.push_back(new Var(0, 0));

    minus.push_back(new Minus());
    add.push_back(new Add());
    square.push_back(new Square());
    //load all nodes
    minus[i] -> load(input_from_outside[i], minus_output[i]);
    add[i] -> load(input[i], minus_output[i], add_output[i]);
    square[i] -> load(add_output[i], square_output[i]);
    superadd -> load_input(square_output[i]);
  }
  superadd -> load_output(layer_output[0]);
}
//pass data forward
void MeanSquareErrorLayer::forward() {
  for (int i = 0; i < neuron; i++) {
    minus[i] -> forward();
  }
  for (int i = 0; i < neuron; i++) {
    add[i] -> forward();
  }
  for (int i = 0; i < neuron; i++) {
    square[i] -> forward();
  }
  superadd -> forward();
  loss_value = layer_output[0] -> data;
}
//pass gradient backward
void MeanSquareErrorLayer::backward() {
  layer_output[0] -> gradient = 1;
  superadd -> backward();

  for (int i = 0; i < neuron; i++) {
    square[i] -> backward();
  }
  for (int i = 0; i < neuron; i++) {
    add[i] -> backward();
  }

  for (int i = 0; i < neuron; i++) {
    minus[i] -> backward();
  }
}
//this layer should receive data from outside
void MeanSquareErrorLayer::load_data_from_outside(std::vector < double > one_data) {
  for (int i = 0; i < one_data.size(); i++) {
    input_from_outside[i] -> data = one_data[i];
  }
}
MeanSquareErrorLayer::~MeanSquareErrorLayer() {
  delete layer_output[0];
  delete superadd;
  for (int i = 0; i < neuron; i++) {
    delete input_from_outside[i];
    delete input[i];
    input[i] = nullptr;
    delete add_output[i];
    delete minus_output[i];
    delete square_output[i];
    delete minus[i];
    delete add[i];
    delete square[i];
  }
}

//================================================================
CrossEntropyLossLayer::CrossEntropyLossLayer(int n): neuron(n) {
  superadd = new SuperAdd();
  minus = new Minus();
  layer_output.push_back(new Var(0, 0));
  superadd_output.push_back(new Var(0, 0));
  for (int i = 0; i < n; i++) {
    //create nodes
    input.push_back(new Var(0, 0));
    input_from_outside.push_back(new Var(0, 0));
    ln_output.push_back(new Var(0, 0));
    mul_output.push_back(new Var(0, 0));

    mul.push_back(new Mul());
    ln.push_back(new Ln());
    //load all nodes
    ln[i] -> load(input[i], ln_output[i]);
    mul[i] -> load(input_from_outside[i], ln_output[i], mul_output[i]);

    superadd -> load_input(mul_output[i]);
  }
  minus -> load(superadd_output[0], layer_output[0]);
  superadd -> load_output(layer_output[0]);
}
//pass data forward
void CrossEntropyLossLayer::forward() {
  for (int i = 0; i < neuron; i++) {
    ln[i] -> forward();
  }
  for (int i = 0; i < neuron; i++) {
    mul[i] -> forward();
  }
  superadd -> forward();
  minus -> forward();
  loss_value = layer_output[0] -> data;
}
//pass gradient backward
void CrossEntropyLossLayer::backward() {
  layer_output[0] -> gradient = 1;
  superadd -> backward();
  minus -> backward();
  for (int i = 0; i < neuron; i++) {
    mul[i] -> backward();
  }
  for (int i = 0; i < neuron; i++) {
    ln[i] -> backward();
  }
}
//this layer should receive data from outside
void CrossEntropyLossLayer::load_data_from_outside(std::vector < double > one_data) {
  for (int i = 0; i < one_data.size(); i++) {
    input_from_outside[i] -> data = one_data[i];
  }
}
CrossEntropyLossLayer::~CrossEntropyLossLayer() {
  delete layer_output[0];
  delete superadd;
  delete minus;
  for (int i = 0; i < neuron; i++) {
    delete input_from_outside[i];
    delete input[i];
    input[i] = nullptr;
    delete ln_output[i];
    delete mul_output[i];
  }
}
