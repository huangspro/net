#include "Node.h"
#include "Layer.h"
#include "Net.h"
#include<iostream>

using namespace std;

int main(){
  Net N;
  N.add_layer(INPUTLAYER,2);
  N.add_layer(NONLINEARLAYER,2,0,NonlinearLayer::RELU,true);
  N.add_layer(MEANSQUAREERRORLAYER,2);
  N.build();
}
