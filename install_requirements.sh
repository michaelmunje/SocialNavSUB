#!/bin/bash

# Install ninja
conda install ninja -y

# Install llama_cpp_python with system tools
LD=/usr/bin/ld CC=/usr/bin/gcc CXX=/usr/bin/g++ pip install llama_cpp_python==0.2.90

# Install remaining packages
pip install -r requirements.txt
