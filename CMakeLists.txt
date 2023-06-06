cmake_minimum_required(VERSION 2.8)

# Set the project name
project(Food_leftover)

# Find OpenCV library
find_package(OpenCV REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})

# Create executables
# Multiple executables can be defined, one for each task
add_executable(detector src/main.cpp src/Detector.cpp)

# Link against OpenCV library
# Each executable should have its own link 
target_link_libraries(detector ${OpenCV_LIBS})