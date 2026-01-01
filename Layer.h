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

#define NONLINEARLAYER 1
#define INPUTLAYER 2
#define SOFTMAXLAYER 3
#define HIDDENLAYER 4
#define MEANSQUAREERRORLAYER 5
#define learning_ratio -0.0001
#include "Node.h"
#include<vector>
#include<iostream>

//This class is only for produce polymorphism 
class Layer{
public:
  int type; //store the type of the layer
  double loss_value; //this one is for loss layer
  std::vector<Var*> input,layer_output;
  Layer(int a):type(a){}
  ~Layer(){}
  virtual void forward(){}
  virtual void backward(){}
  virtual void train(){}
  virtual void load_data_from_outside(std::vector<double> a){}
  virtual void input_data(std::vector<double> a){}
  virtual void connect_to_last_layer_output(std::vector<Var*> a){};
  virtual void connect_to_next_layer_input(std::vector<Var*> a){}
  virtual void none(){}
};

//This layer is the core layer of a neural net, containing a nonlinear function
class NonlinearLayer : public Layer{
public:
  static const int RELU=1;
  static const int TANH=2;
  static const int SIGMOID=3;
  int neuron;
  int function_type;
  std::vector<Var*> input, layer_output;
  std::vector<Ope*> opes;
  
  NonlinearLayer(int n, int f):neuron(n),function_type(f),Layer(NONLINEARLAYER){
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
    }
  }
  //forward the data
  void forward(){
    std::cout<<"okok: "<<input[0]->gradient<<" koko"<<std::endl;
    for(auto i=opes.begin();i!=opes.end();i++){
      (*i)->forward();
    }
  }
  //backward the gradient
  void backward(){
    for(auto i=opes.begin();i!=opes.end();i++){
      (*i)->backward();
    }
  }
  //a nonlinear layer can should have the ability to connect with the last or the next layer
  void connect_to_last_layer_output(std::vector<Var*> last_layer_output){
    for(int i=0; i<last_layer_output.size(); i++){
      delete input[i];
      input[i]=last_layer_output[i];
      opes[i]->load(input[i],layer_output[i]);
    }
  }
  void connect_to_next_layer_input(std::vector<Var*>& next_layer_input){
    for(int i=0; i<next_layer_input.size(); i++){
      delete next_layer_input[i];
      next_layer_input[i]=layer_output[i];
      opes[i]->load(input[i],layer_output[i]);
    }
  }
  //unload, noted that the input and output layer is connected to other, so they should be deleted casually
  ~NonlinearLayer(){
    for(int i=0;i<neuron;i++){
     delete input[i];
     input[i]=nullptr;
     delete layer_output[i];
     layer_output[i]=nullptr;
     delete opes[i];
    }
  }
};

//This layer is an input layer, which can tackle data input
class InputLayer : public Layer{
public:
  int neuron; 
  std::vector<Var*> input, weight, bias, mul_output, add_output, layer_output;
  std::vector<Ope*> mul, add;
  void none(){}
  InputLayer(int n):neuron(n),Layer(INPUTLAYER){
    for(int i=0;i<neuron;i++){
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
    }
  }
  //forward the data
  void forward(){
    for(auto i=mul.begin();i!=mul.end();i++){
      (*i)->forward();
    }
    for(auto i=add.begin();i!=add.end();i++){
      (*i)->forward();
    }
  }
  //backward the gradient
  void backward(){
    
    
    for(auto i=add.begin();i!=add.end();i++){
      (*i)->backward();
    }
    for(auto i=mul.begin();i!=mul.end();i++){
      (*i)->backward();
    }
  }
  //train the layer
  void train(){
    for(int i=0;i<neuron;i++){
      weight[i]->data+=learning_ratio*weight[i]->gradient;
      bias[i]->data+=learning_ratio*bias[i]->gradient;
    }
  }
  //this layer can receive data outside
  void input_data(std::vector<double> one_data){
    for(int i=0;i<one_data.size();i++){
      input[i]->data=one_data[i];
    }
  }
  ~InputLayer(){
    for(int i=0;i<neuron;i++){
     delete input[i];
     input[i]=nullptr;
     delete weight[i];
     delete bias[i];
     delete mul_output[i];
     delete add_output[i];
     delete layer_output[i];
     layer_output[i]=nullptr;
     delete mul[i];
     delete add[i];
    }
  }
};

//This layer is a hiddenlayer, containing two dimensional vector to store the weights for each unit
class HiddenLayer : public Layer{
public:
  int neuron;
  int last_layer_neuron_number;
  std::vector<Var*> input,bias,layer_output;
  std::vector<std::vector<Var*>> weights,mul_output;
  std::vector<Ope*> superadd;
  std::vector<std::vector<Ope*>>mul; 
  void none(){}
  HiddenLayer(int n, int last_layer):Layer(HIDDENLAYER),neuron(n),last_layer_neuron_number(last_layer){
    //layer
    //firstly, the single one dimension part
    for(int i=0;i<n;i++){
      bias.push_back(new Var(0,0));
      layer_output.push_back(new Var(0,0));
      superadd.push_back(new SuperAdd());
    }
    for(int i=0;i<last_layer;i++){
      input.push_back(new Var(0,0));
    }
    //the two dimension part
    for(int i=0;i<n;i++){
      std::vector<Var*> tem_mul_output(last_layer);
      std::vector<Var*> tem_weights(last_layer);
      std::vector<Ope*> tem_mul(last_layer);
      for(int ii=0;ii<last_layer;ii++){
        tem_mul_output[ii]=new Var(0,0);
        tem_weights[ii]=new Var(0,0);
        tem_mul[ii]=new Mul();
        //load the multiply nodes
        tem_mul[ii]->load(tem_weights[ii],input[ii] , tem_mul_output[ii]);
      }

      mul_output.push_back(tem_mul_output);
      weights.push_back(tem_weights);
      mul.push_back(tem_mul);
    }
    //load the superadd operation nodes
    for(int i=0;i<n;i++){
      for(int ii=0;ii<last_layer;ii++){
        superadd[i]->load_input(mul_output[i][ii]);
        superadd[i]->load_output(layer_output[i]);
      }
    }
  }
  //forward the data
  void forward(){
    for(int i=0;i<neuron;i++){
      for(int ii=0;ii<last_layer_neuron_number;ii++){
        mul[i][ii]->forward();
      }
    }
    for(int i=0;i<neuron;i++){
      superadd[i]->forward();
    }
  }
  //pass backward the gradient
  void backward(){
    for(int i=0;i<neuron;i++){
      superadd[i]->backward();
    }
    for(int i=0;i<neuron;i++){
      for(int ii=0;ii<last_layer_neuron_number;ii++){
        mul[i][ii]->backward();
      }
    }
  }
  //train the layer
  void train(){
    for(int i=0;i<bias.size();i++){
      bias[i]->data+=learning_ratio*bias[i]->gradient;
    }
    for(int i=0;i<weights.size();i++){
      for(int ii=0;ii<weights[i].size();ii++){
        weights[i][ii]->data+=learning_ratio*weights[i][ii]->gradient;
      }
    }
  }
  //to avoid delete null pointer, input and layer_output should be nullptr after being deleted
  ~HiddenLayer(){
    for(int i=0;i<last_layer_neuron_number;i++){
    delete input[i];
    input[i]=nullptr;
    delete superadd[i];
    delete layer_output[i];
    layer_output[i]=nullptr;
    delete bias[i];
      for(int ii=0;ii<last_layer_neuron_number;ii++){
        delete mul[i][ii];
        delete mul_output[i][ii];
        delete weights[i][ii];
      }
    }
  }
};

//This layer is for calculate the softmax function output
class SoftmaxLayer : public Layer{
public:  
  int neuron;
  std::vector<Var*> input, e_output, layer_output;
  Var* superadd_output, *dev_output;
  std::vector<Ope*> exp, mul;
  Ope* superadd, *dev;
  void none(){}
  SoftmaxLayer(int n):neuron(n),Layer(SOFTMAXLAYER){
    //create nodes
    superadd_output=new Var(0,0);
    dev_output=new Var(0,0);
    superadd=new SuperAdd();
    dev=new Dev();
    //load all the nodes
    superadd->load_output(superadd_output);
    dev->load(superadd_output,dev_output);
    for(int i=0;i<n;i++){
      input.push_back(new Var(0,0));
      e_output.push_back(new Var(0,0));
      layer_output.push_back(new Var(0,0));
      exp.push_back(new Exp());
      mul.push_back(new Mul());
      //load all the nodes
      mul[i]->load(dev_output,e_output[i],layer_output[i]);
      exp[i]->load(input[i],e_output[i]);
      superadd->load_input(e_output[i]);
    }
  }
  //pass forward data
  void forward(){
    for(int i=0;i<neuron;i++){
      exp[i]->forward();
    }
    superadd->forward();
    dev->forward();
    for(int i=0;i<neuron;i++){
      mul[i]->forward();
    }
  }
  //pass the gradient backward
  void backward(){
    for(int i=0;i<neuron;i++){
      mul[i]->backward();
    }
    dev->backward();
    superadd->backward();
    for(int i=0;i<neuron;i++){
      exp[i]->backward();
    }
  }
  //a Softmax layer can should have the ability to connect with the last or the next layer
  void connect_to_last_layer_output(std::vector<Var*> last_layer_output){
    for(int i=0; i<last_layer_output.size(); i++){
      delete input[i];
      input[i]=last_layer_output[i];
    }
  }
  void connect_to_next_layer_input(std::vector<Var*> next_layer_input){
    for(int i=0; i<next_layer_input.size(); i++){
      delete next_layer_input[i];
      next_layer_input[i]=layer_output[i];
    }
  }
  ~SoftmaxLayer(){
    delete superadd_output;
    delete dev_output;
    delete superadd;
    delete dev;
    for(int i=0;i<neuron;i++){
      delete mul[i];
      delete input[i];
      input[i]=nullptr;
      delete e_output[i];
      delete layer_output[i];
      layer_output[i]=nullptr;
    }
  }
};

//MeanSquareErrorLayer
class MeanSquareErrorLayer : public Layer{
public:
  int neuron;
  std::vector<Var*> input,input_from_outside,minus_output,add_output,square_output,layer_output;
  std::vector<Ope*> minus,add,square;
  Ope* superadd;
  void none(){}
  MeanSquareErrorLayer(int n):neuron(n),Layer(MEANSQUAREERRORLAYER){
    superadd=new SuperAdd();
    layer_output.push_back(new Var(0,0));
    for(int i=0;i<n;i++){
      //create nodes
      input.push_back(new Var(0,0));
      input_from_outside.push_back(new Var(0,0));
      minus_output.push_back(new Var(0,0));
      add_output.push_back(new Var(0,0));
      square_output.push_back(new Var(0,0));

      minus.push_back(new Minus());
      add.push_back(new Add());
      square.push_back(new Square());
      //load all nodes
      minus[i]->load(input_from_outside[i], minus_output[i]);
      add[i]->load(input[i], minus_output[i],add_output[i]);
      square[i]->load(add_output[i],square_output[i]);
      superadd->load_input(square_output[i]);
    }
    superadd->load_output(layer_output[0]);
  }
  //pass data forward
  void forward(){
    for(int i=0;i<neuron;i++){
      minus[i]->forward();
    } 
    for(int i=0;i<neuron;i++){
      add[i]->forward();
    } 
    for(int i=0;i<neuron;i++){
      square[i]->forward();
    } 
    superadd->forward();
    loss_value=layer_output[0]->data;
  }
  //pass gradient backward
  void backward(){
    layer_output[0]->gradient=1;
    superadd->backward();
    
    for(int i=0;i<neuron;i++){
      square[i]->backward();
    } 
    for(int i=0;i<neuron;i++){
      add[i]->backward();
    } 
    
    for(int i=0;i<neuron;i++){
      minus[i]->backward();
    } 
    
  }
  //this layer should receive data from outside
  void load_data_from_outside(std::vector<double> one_data){
    for(int i=0;i<one_data.size();i++){
      input_from_outside[i]->data=one_data[i];
    }
  }
  ~MeanSquareErrorLayer(){
    delete layer_output[0];
    delete superadd;
    for(int i=0;i<neuron;i++){
      delete input_from_outside[i];
      delete input[i];
      input[i]=nullptr;
      delete add_output[i];
      delete minus_output[i];
      delete square_output[i];
      delete minus[i];
      delete add[i];
      delete square[i];
    } 
  }
};


#endif
