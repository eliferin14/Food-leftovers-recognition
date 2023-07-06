cmake_minimum_required(VERSION 2.8)

# Set the project name
project(Food_leftover)

# Find OpenCV library
find_package(OpenCV REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})

# Create executables
# Multiple executables can be defined, one for each task
add_executable(detector src/main.cpp src/Detector.cpp)
add_executable(test src/test.cpp src/Detector.cpp)
add_executable(boundingBox src/BBtest.cpp src/BoundBox.cpp src/Detector.cpp src/Crisci.cpp)
add_executable(metrics_test src/metrics/metrics_test.cpp src/metrics/DatasetLoader.cpp)

# Link against OpenCV library
# Each executable should have its own link 
target_link_libraries(detector ${OpenCV_LIBS})
target_link_libraries(test ${OpenCV_LIBS})
target_link_libraries(boundingBox ${OpenCV_LIBS})
target_link_libraries(metrics_test ${OpenCV_LIBS})