#include "Node.h"
#include<iostream>

using namespace std;

int main(){
  vector<Ope*> allopes;
  
  Var* root=new Var(1,0);
  Ope* o1=new Minus();
  Var* v1=new Var(1,0);
  o1->load(root,v1);
  vector<Ope*> tem;
  tem.push_back(o1);
  root->load(tem);
  
  allopes.push_back(o1);
  for(auto i=allopes.begin();i!=allopes.end();i++){
      (*i)->forward();
  }
  cout<<v1->data;
}
