#ifndef _NODE_H_
#define _NODE_H_  // ← Fixed: was missing #define
#include<cmath>

class Node {
public:
    Node() {}
    virtual double forward();
    virtual void backward();
    virtual ~Node() {}
};

class Var {
public:
double gradient;
    double data;
    Var(double d, double g = 0.0) : data(d), gradient(g) {}
};

//=====================================================================================

class Add : public Node {
public:
    Add() : Node() {}
    double forword(Var* a, Var* b) {
        return a->data+ b->data;
    }
    void backward(Var* a, Var* b, Var* c) {
        a->gradient= c->gradient;
        b->gradient= c->gradient;
    }
}; 

class Mul : public Node {
public:
    Mul() : Node() {}
    double forword(Var* a, Var* b) {
        return a->data* b->data;
    }
    void backward(Var* a, Var* b, Var* c) {
        a->gradient= c->gradient* b->data;
        b->gradient= c->gradient* a->data;
    }
};  
class Dev : public Node { // this Node is for 1/... (as per your comment)
public:
    Dev() : Node() {}
    double forword(Var* a) {
        return 1/a->data;
    }
    void backward(Var* a,Var* c) {
        a->gradient=c->gradient*( 1/(a->data*a->data*(-1)));
    }
};

class Minus : public Node { // this Node is for 1/... (as per your comment)
public:
    Minus() : Node() {}
    double forword(Var* a) {
        return -1*a->data;
    }
    void backward(Var* a,Var* c) {
        a->gradient= c->gradient*(-1);
    }
};

class Relu : public Node {
public:
    Relu() : Node() {}
    double forword(Var* a) {
        return a->data > 0 ? a->data : 0;
    }
    void backward(Var* a, Var* c) {
        a->gradient = c->gradient * (a->data > 0 ? 1.0 : 0.0);
    }
};

class Sigmoid : public Node {
public:
    Sigmoid() : Node() {}
    double forword(Var* a) {
        return 1.0 / (1.0 + exp(-a->data));
    }
    void backward(Var* a, Var* c) {
        double s = 1.0 / (1.0 + exp(-a->data));
        a->gradient = c->gradient * (s * (1.0 - s));
    }
};

class Tanh : public Node {
public:
    Tanh() : Node() {}
    double forword(Var* a) {
        return tanh(a->data);
    }
    void backward(Var* a, Var* c) {
        double t = tanh(a->data);
        a->gradient = c->gradient * (1.0 - t * t);
    }
};

class Exp : public Node {
public:
    Exp() : Node() {}
    double forword(Var* a) {
        return exp(a->data);
    }
    void backward(Var* a, Var* c) {
        a->gradient = c->gradient * exp(a->data);
    }
};

class Ln : public Node {
public:
    Ln() : Node() {}
    double forword(Var* a) {
        return log(a->data);
    }
    void backward(Var* a, Var* c) {
        a->gradient = c->gradient * (1.0 / a->data);
    }
};

class Pow : public Node {
public:
    double exponent;
    Pow(double e) : Node(), exponent(e) {}
    double forword(Var* a) {
        return pow(a->data, exponent);
    }
    void backward(Var* a, Var* c) {
        // d(x^n)/dx = n * x^(n-1)
        a->gradient += c->gradient * (exponent * pow(a->data, exponent - 1));
    }
};

class Sqrt : public Node {
public:
    Sqrt() : Node() {}
    double forword(Var* a) {
        return sqrt(a->data);
    }
    void backward(Var* a, Var* c) {
        // d(sqrt(x))/dx = 1 / (2 * sqrt(x))
        a->gradient += c->gradient * (1.0 / (2.0 * sqrt(a->data)));
    }
};

class SquareDiff : public Node { //  (a-b)^2
public:
    SquareDiff() : Node() {}
    double forword(Var* a, Var* b) {
        double diff = a->data - b->data;
        return diff * diff;
    }
    void backward(Var* a, Var* b, Var* c) {
        double diff = a->data - b->data;
        a->gradient += c->gradient * (2.0 * diff);
        b->gradient += c->gradient * (-2.0 * diff);
    }
};
#endif  
