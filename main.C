#include "Node.h"
#include<iostream>

using namespace std;

int main(){
  vector<Ope*> o;
  vector<Var*> v;
  for(int i=0;i<9;i++){v.push_back(new Var(0,0));}
  o.push_back(new Dev());
  o.push_back(new Minus());
  o.push_back(new Dev());
  o.push_back(new Add());
  o.push_back(new Add());
  o.push_back(new Ln());
  
  v[0]->load(o[0]);
  v[1]->load(o[1]);
  v[2]->load(o[2]);
  v[4-1]->load(o[3]);
  v[5-1]->load(o[3]);
  v[7-1]->load(o[4]);
  v[5]->load(o[4]);
  v[7]->load(o[5]);
  v[8]->load(nullptr);
  
  o[0]->load(v[0],v[3]);
  o[1]->load(v[1],v[4]);
  o[2]->load(v[2],v[6]);
  o[3]->load(v[3],v[4],v[5]);
  o[4]->load(v[5],v[6],v[7]);
  o[5]->load(v[7],v[8]);
  
  v[0]->data=1.9;
  v[1]->data=-1.6;
  v[2]->data=0.3;
  v[8]->gradient=0.36;
  for(auto i=o.begin();i!=o.end();i++){
      (*i)->forward();
  }
  cout<<v[8]->data<<endl;
  for(auto i=o.rbegin();i!=o.rend();i++){
      (*i)->backward();
  }
  cout<<v[0]->gradient<<"  "<<v[1]->gradient<<"  "<<v[2]->gradient<<endl;
}
