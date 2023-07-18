#ifndef IOU
#define IOU

#include <opencv2/opencv.hpp>
#include "../utils.hpp"

void getUnionMaskFromBB(std::vector<cv::Rect>& boundingBoxes, cv::Mat& mask);

double iou_twoMasks(cv::Mat& mask, cv::Mat& trueMask);
double iou_twoImagesUnionBB(std::vector<cv::Rect>& bb, std::vector<cv::Rect>& trueBB);

#endif