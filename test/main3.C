#include "../base_class/ConvolutionLayer.h"
#include<iostream>
#include<vector>

using namespace std;

int main(){
  ConvolutionLayer C(3,3,4,4,1);
  vector<vector<double>> testdata ={{1,2,3,4},{2,3,4,5},{3,4,5,6},{4,5,6,7}};
  C.load(testdata);
  C.forward();

  for(int i=0;i<C.output.size();i++){
    for(int ii=0;ii<C.output[0].size();ii++){
      cout<<C.output[i][ii]->data<<" ";
      C.output[i][ii]->gradient=1;
    }
    cout<<endl;
  }  
  C.backward();
  for(int i=0;i<C.output.size();i++){
    for(int ii=0;ii<C.output[0].size();ii++){
      cout<<C.g2(i,ii)->gradient<<" ";
    }
    cout<<endl;
  } 
}
