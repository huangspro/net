#!/bin/bash

g++ -w -o main main.C && ./main

echo "clear?"
read -r i
if [ -z "$i" ]; then
    clear
fi
