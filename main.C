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
  
  v1->gradient=1;
  allopes.push_back(o1);
  for(auto i=allopes.begin();i!=allopes.end();i++){
      (*i)->forward();
  }
  for(auto i=allopes.rbegin();i!=allopes.rend();i++){
      (*i)->backward();
  }
  
  cout<<root->gradient<<endl;
}
