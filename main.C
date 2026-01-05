#include "Node.h"
#include "Layer.h"
#include "Net.h"
#include<iostream>

using namespace std;

int main(){
  InputLayer* I=new InputLayer(2);
  NonlinearLayer* N1=new NonlinearLayer(2, NonlinearLayer::TANH);
  MeanSquareErrorLayer* M=new MeanSquareErrorLayer(1);
  NonlinearLayer* N2=new NonlinearLayer(1, NonlinearLayer::TANH);
  NonlinearLayer* N3=new NonlinearLayer(1, NonlinearLayer::LINEAR);
  HiddenLayer* H1=new HiddenLayer(1,2);
  HiddenLayer* H2=new HiddenLayer(1,1);
  
  N1->connect_to_last_layer_output(I->layer_output);
  N1->connect_to_next_layer_input(H1->input);
  N2->connect_to_last_layer_output(H1->layer_output);
  N2->connect_to_next_layer_input(H2->input);
  N3->connect_to_last_layer_output(H2->layer_output);
  N3->connect_to_next_layer_input(M->input);
  
  vector<vector<double>> inputdata={{160,45},{165,55},{170,60},{172,65},{175,70},{168,72},{180,80},{158,50},{162,68},{170,85}};
  vector<vector<double>> testdata={{17.6},{20.2},{20.8},{22.0},{22.9},{25.5},{24.7},{20.0},{25.9},{29.4}};
  
  double last_loss;
  int i=0;
  while(true){
    i++;
    last_loss=0;
    for(int ii=0;ii<10;ii++){
      M->load_data_from_outside(testdata[ii]);
      I->input_data(inputdata[ii]);
      
      I->forward();
      N1->forward();
      H1->forward();
      N2->forward();
      H2->forward();
      N3->forward();
      M->forward();
      
      M->backward();
      N3->backward();
      H2->backward();
      N2->backward();
      H1->backward();
      N1->backward();
      I->backward();
      
      I->train();
      H1->train();
      H2->train();
      
      last_loss+=M->layer_output[0]->data;
    }
    if(i%100000==0){//cout<<"loss: "<<last_loss/10<<endl;
      cout<<I->weight[0]->gradient<<" "<<I->weight[1]->gradient<<" "<<H1->weights[0][0]->gradient<<" "<<H1->weights[0][1]->gradient<<endl;
      cout<<H2->weights[0][0]->gradient<<endl;
    }
  }

}
