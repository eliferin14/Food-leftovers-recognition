#ifndef BOUNDING_BOXES
#define BOUNDING_BOXES

#include <opencv2/opencv.hpp>
#include <fstream>

void getBoundingBoxes(std::vector<std::vector<cv::Point2f>>& clusters, std::vector<cv::Rect>& boundingBoxes);
void extractPlatesBB(cv::Mat& src, std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& plates);

void saveBoundingBoxes(std::vector<cv::Rect>& boundingBoxes, std::vector<int>& labels, std::string filepath);

#endif