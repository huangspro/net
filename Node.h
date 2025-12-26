#ifndef _NODE_H_
#endif _NODE_H_
class Node{
public:
  Node():gradient(0){}
  virtual double forward();
  virtual void backward();
  virtual ~Node(){}
};

class Var{
private:
  double gradient;
  double data;
public:
  Node(double g, double d):gradient(g),data(d){}
};
//=====================================================================================

class Add : public Node{
public:
  Add():Node(){}
  double forword(Var* a, Var* b){
    return a->data+b->data;
  }
  void backward(Var* a, Var* b, Var* c){
    a->gradient=c->gradient;
    b->gradient=c->gradient;
};

class Mul : public Node{
public:
  Mul():Node(){}
  double forword(Var* a, Var* b){
    return a->data*b->data;
  }
  void backward(Var* a, Var* b, Var* c){
    a->gradient=c->gradient*b->data;
    b->gradient=c->gradient*a->data;
};

class Dev : public Node{ //this Node is for 1/...
public:
  Dev():Node(){}
  double forword(Var* a, Var* b){
    return a->data*b->data;
  }
  void backward(Var* a, Var* b, Var* c){
    a->gradient=c->gradient*b->data;
    b->gradient=c->gradient*a->data;
};
#endif
