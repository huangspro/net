#!/bin/bash

g++ -w -o main"$1" main$1.C ../base_class/Layer.C ../base_class/ConvolutionLayer.C && ./main$1

