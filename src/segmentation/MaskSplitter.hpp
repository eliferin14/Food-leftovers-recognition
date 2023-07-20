#ifndef MASK_SPLITTER
#define MASK_SPLITTER

#include <opencv2/opencv.hpp>
#include "../utils.hpp"

void splitMask(cv::Mat& inputMask, std::vector<cv::Mat>& outputMasks);

#endif