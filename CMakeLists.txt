cmake_minimum_required(VERSION 2.8)

# Set the project name
project(Food_leftover)

# Find OpenCV library
find_package(OpenCV REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})

# Segmentation
add_executable(segmentation src/segmentation/segmentation.cpp src/segmentation/MeanShift.cpp src/utils.cpp src/segmentation/BoundingBoxes.cpp src/segmentation/Masks.cpp src/segmentation/MaskSplitter.cpp)
target_link_libraries(segmentation ${OpenCV_LIBS})

# Metrics
add_executable(metrics src/metrics/metrics.cpp src/metrics/DatasetLoader.cpp src/metrics/MeanAveragePrecision.cpp)
target_link_libraries(metrics ${OpenCV_LIBS})

# Gianni-Crisci
add_executable(gianniCrisci src/gianniCrisci.cpp src/Detector.cpp src/BoundBox.cpp src/Crisci.cpp src/Seg.cpp)
target_link_libraries(gianniCrisci ${OpenCV_LIBS})

# Create executables
# Multiple executables can be defined, one for each task
add_executable(detector src/main.cpp src/Detector.cpp src/utils.cpp)
#add_executable(test src/test.cpp src/Detector.cpp)
#add_executable(boundingBox src/BBtest.cpp src/BoundBox.cpp src/Detector.cpp src/Crisci.cpp)
#add_executable(metrics_test src/metrics/metrics_test.cpp src/metrics/DatasetLoader.cpp src/metrics/MeanAveragePrecision.cpp)
#add_executable(edge_test src/edge/edgeTest.cpp src/edge/EdgeSegmentation.cpp src/Detector.cpp)

# Link against OpenCV library
# Each executable should have its own link 
target_link_libraries(detector ${OpenCV_LIBS})
#target_link_libraries(test ${OpenCV_LIBS})
#target_link_libraries(boundingBox ${OpenCV_LIBS})
#target_link_libraries(metrics_test ${OpenCV_LIBS})
#target_link_libraries(edge_test ${OpenCV_LIBS})
