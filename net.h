#ifndef _NET_H_
#define _NET_H_

#include "Node.h"
#include<vector>
#include<iostream>

class NonlinearLayer{
public:
  static const int RELU=1;
  static const int TANH=2;
  static const int SIGMOID=3;
  int neuron;
  int function_type;
  std::vector<Var*> input, layer_output;
  std::vector<Ope*> opes;
  
  NonlinearLayer(int n, int f):neuron(n),function_type(f){
    for(int i=0;i<n;i++){
     //create all nodes
      input.push_back(new Var(0,0));
      layer_output.push_back(new Var(0,0));
      switch(f){//judge nonlinear function type
        case RELU:
          opes.push_back(new Relu());break;
        case TANH:
          opes.push_back(new Tanh());break;
        case SIGMOID:
          opes.push_back(new Sigmoid());break;
      }
      //load all operation nodes
      opes[i]->load(input[i],layer_output[i]);
      //load all variable nodes
      input[i]->load(opes[i]);
      layer_output[i]->load(opes[i]);
    }
  }
  void connect_to_next_layer(){
  
  }
  
  ~NonlinearLayer(){
    for(int i=0;i<neuron;i++){
     delete input[i];
     delete layer_output[i];
     delete opes[i];
    }
  }
};


class InputLayer{
public:
  int neuron; 
  std::vector<Var*> input, weight, bias, mul_output, add_output, layer_output;
  std::vector<Ope*> mul, add;
  
  InputLayer(int n):neuron(n){
  
    for(int i=0;i<neuron;i++){
      std::cout<<i<<std::endl;
      //create the Nodes
      input.push_back(new Var(0,0));
      weight.push_back(new Var(0,0));
      bias.push_back(new Var(0,0));
      layer_output.push_back(new Var(0,0));
      mul_output.push_back(new Var(0,0));
      add_output.push_back(new Var(0,0));
      
      mul.push_back(new Mul());
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
      layer_output[i]->load(nullptr);
    }
  }
  
  void connect_to_nonlinear_layer(NonlinearLayer* next){
    for(int i=0;i<neuron;i++){
      delete next->input[i];
      next->input[i]=layer_output[i];
      next->input[i]->load(next->opes[i]);
      next->opes[i]->load(next->input[i],next->layer_output[i]);
    }
  }
  
  void forward(){
    for(auto i=mul.begin();i!=mul.end();i++){
      (*i)->forward();
    }
    for(auto i=add.begin();i!=add.end();i++){
      (*i)->forward();
    }
  }
  
  ~InputLayer(){
    for(int i=0;i<neuron;i++){
     delete input[i];
     delete weight[i];
     delete bias[i];
     delete mul_output[i];
     delete add_output[i];
     delete layer_output[i];
     delete mul[i];
     delete add[i];
    }
  }
};

class HiddenLayer{
public:
  int neuron;
  int last_layer_neuron_number;
  vector<Var*> input,bias,layer_output;
  vector<vector<Var*>> weights,mul_output;
  vector<Ope*> superadd,add;
  vector<vector<Ope*>>mul; 
  HiddenLayer(int n, int last_layer):neuron(n),last_layer_neuron_number(last){
    //create
    for(int i=0;i<n;i++){
      bias[i]=new Var(0,0);
      layer_output[i]=new Var(0,0);
      superadd[i]=new Ope();
      add[i]=new Ope();
    }
    for(int i=0;i<last_layer;i++){
      input[i]=new Var(0,0);
    }
    for(int i=0;i<n;i++){
      vector<Var*> tem_mul_output(last_layer);
      vector<Var*> tem_weights(last_layer);
      vector<Ope*> tem_mul(last_layer);
      for(int ii=0;ii<last_layer;ii++){
        tem_mul_output[ii]=new Var(0,0);
        tem_weights[ii]=new Var(0,0);
        tem_mul[ii]=new Ope();
      }
      mul_output.push_back(tem_mul_output);
      weights.push_back(tem_weights);
      mull.push_back(tem_mul);
    }
    //load
    for(int i=0;i<n;i++){
      bias[i]->load(add[i]);
      layer_output[i]->load(nullptr);
      vector<Var*> tem(n);
      for(int ii=0;ii<last_layer;ii++)tem.push_back(mul_output[ii]);
      superadd[i]->load(tem);
      add[i]->load(superadd[i],bias[i],layer_output[i]);
    }
    for(int i=0;i<n;i++){
      for(int ii=0;ii<last_layer;ii++){
        input[i]->load(mul[i]);
      }
    }
  }
};
#endif
