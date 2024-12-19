cmake_minimum_required(VERSION 3.10)
project(OpenCL-Dev-Realtime-Image-Processing VERSION 0.1.0 LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# Add the source subdirectory
SUBDIRS(src)