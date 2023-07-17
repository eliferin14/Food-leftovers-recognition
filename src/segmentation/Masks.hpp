#ifndef MASKS
#define MASKS

#include <opencv2/opencv.hpp>

void grabCutSegmentation(cv::Mat& src, std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& masks);
void removeLowSaturation(cv::Mat& src, cv::Mat& dst, double threshold);
void masksPostprocess(std::vector<cv::Mat>& masks);
void refineBoundingBoxes(std::vector<cv::Mat>& masks, std::vector<cv::Rect>& boundingBoxes);

#endif