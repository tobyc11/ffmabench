cmake_minimum_required(VERSION 3.20)

project(bpttrap CXX CUDA)

add_executable(ffmabench ffmabench.cu)
set_property(TARGET ffmabench PROPERTY CUDA_ARCHITECTURES 60 61 62 70 72 75 80 86)
