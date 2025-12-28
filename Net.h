#ifndef _NET_H_
#define _NET_H_

#include "Layer.h"
#include<vector>
#define NONLINEARLAYER 1
#define INPUTLAYER 2
#define SOFTMAXLAYER 3
#define HIDDENLAYER 4
#define MEANSQUAREERRORLAYER 5
class Net{
public:
  std::vector<Layer*> AllLayer;
  Net(){
  }
  //add a layer, noted that only hidden layer should pass the last_layer_neuron
  void add_layer(int layer_type, int neuron, int last_layer_neuron = 0){
    switch(layer_type){
      case INPUTLAYER:
          Layer* newInputLayer = new InputLayer(neuron);
          break;
    }
  }
};

#endif
