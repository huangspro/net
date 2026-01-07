#!/bin/bash

g++ -w -o main"$1" main$1.C base_class/Layer.C && ./main$1

