cmake_minimum_required(VERSION 2.8)

# Set the project name
project(Food_leftover)

# Find OpenCV library
find_package(OpenCV REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})

# Segmentation
add_executable(segmentation src/segmentation/segmentation.cpp src/segmentation/MeanShift.cpp src/utils.cpp src/segmentation/BoundingBoxes.cpp src/segmentation/Masks.cpp src/segmentation/MaskSplitter.cpp src/segmentation/Estimator.cpp)
target_link_libraries(segmentation ${OpenCV_LIBS})

# Metrics
add_executable(metrics src/metrics/metrics.cpp src/metrics/DatasetLoader.cpp src/metrics/IntersectionOverUnion.cpp src/metrics/MeanAveragePrecision.cpp src/utils.cpp)
target_link_libraries(metrics ${OpenCV_LIBS})

