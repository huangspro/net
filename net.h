#include "Node.h"
#include<vector>

class InputLayer{
public:
  int neuron; //number of neuron in this layer
  std::vector<Var*> input, weight, bias, mul_output, add_output, layer_output;
  std::vector<Ope*> mul, add;
  InputLayer(int n):neuron(n){
    for(int i=0;i<neuron;i++){
      //create the Nodes
      input.push_back(new Var(0,0));
      weight.push_back(new Var(0,0));
      bias.push_back(new Var(0,0));
      layer_output.push_back(new Var(0,0));
      mul_output.push_back(new Var(0,0));
      add_output.push_back(new Var(0,0));
      
      mul.push_back(new Mull());
      add.push_back(new Add());
      //load the operation nodes
      mul[i]->load(input[i],weight[i],mul_output[i]);
      add[i]->load(mul_output[i],bias[i],layer_output[i]);
      //load the variable nodes
      input[i]->load(mul[i]);
      weight[i]->load(mul[i]);
      mul_output[i]->load(add[i]);
      bias[i]->load(add[i]);
      //load the output layer
      layer_output(nullptr);
    }
  }
  
  void connect_to_next_layer(){
    
  }
  
  void forward(){
    for(auto i=mul.begin();i!=mul.end();i++){
      (*i)->forward();
    }
    for(auto i=add.begin();i!=add.end();i++){
      (*i)->forward();
    }
  }
};
