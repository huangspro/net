#ifndef _NET_H_
#define _NET_H_

#include "Layer.h"
#include<iostream>
#include<vector>

class Net{
public:
  std::vector<Layer*> AllLayer;
  int mark_output; //this variable is for storing the index of the output layer in the net, especially, in the AllLayer
  Net():mark_output(-1){}
  //add a layer, noted that only hidden layer should pass the last_layer_neuron
  //mark is for marking the output layer, because we don't know where is the output layer is
  void add_layer(int layer_type, int neuron, int last_layer_neuron = 0,int function_type=0, int mark=false){
    Layer* newLayer;
    if(mark)mark_output=AllLayer.size();
    switch(layer_type){
      case INPUTLAYER:
          newLayer = new InputLayer(neuron);
          AllLayer.push_back(newLayer);
          break;
      case NONLINEARLAYER:
          newLayer=new NonlinearLayer(neuron,function_type);//we need a function type here, such as NonlinearLayer::RELU
          AllLayer.push_back(newLayer);
          break;
      case SOFTMAXLAYER:
          newLayer = new SoftmaxLayer(neuron);
          AllLayer.push_back(newLayer);
          break;
      case MEANSQUAREERRORLAYER:
          newLayer=new MeanSquareErrorLayer(neuron);
          AllLayer.push_back(newLayer);
          break;
      case HIDDENLAYER:
          newLayer=new HiddenLayer(neuron,last_layer_neuron);//we need the number of neuron of the last layer
          AllLayer.push_back(newLayer);
          break;
    }
  }
  
  //this function is to connect all the layers
  //here we suppose that the first layer is input layer, and the last layer is loss layer, and the middle is hidden layer and nonlinear layer
  void build(){
    for(int i=0;i<AllLayer.size();i++){
      if(AllLayer[i]->type==NONLINEARLAYER || AllLayer[i]->type==SOFTMAXLAYER){
        AllLayer[i]->connect_to_last_layer_output(AllLayer[i-1]->input);
        AllLayer[i]->connect_to_next_layer_input(AllLayer[i+1]->layer_output);
      }
    }
  }
  void forward(){
    for(int i=0;i<AllLayer.size();i++){
      AllLayer[i]->forward();
    }
  }
  //get the lost
  double loss(std::vector<double> input, std::vector<double> train_data){
    AllLayer[0]->input_data(input);
    AllLayer.back()->load_data_from_outside(train_data);
    forward();
    return AllLayer.back()->loss_value;
  }
  //calculate output of the net given data in vector form
  std::vector<double> output(std::vector<double> input){
    AllLayer[0]->input_data(input);
    forward();
    std::vector<Var*> tem=AllLayer[mark_output]->layer_output;
    std::vector<double> result(tem.size());
    for(int i=0;i<tem.size();i++){
      result.push_back(tem[i]->data);
    }
    return result;
  }
};

#endif
