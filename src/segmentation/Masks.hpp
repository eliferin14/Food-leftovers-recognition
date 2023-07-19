#ifndef MASKS
#define MASKS

#include <opencv2/opencv.hpp>

void grabCutSegmentation(cv::Mat& src, std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& masks);
void removeLowSaturation(cv::Mat& src, cv::Mat& dst, double threshold);
void removeLowSaturation_otsu(cv::Mat& src, cv::Mat& dst);
void masksPostprocess(std::vector<cv::Mat>& masks, std::string filename);
void refineBoundingBoxes(std::vector<cv::Mat>& masks, std::vector<cv::Rect>& boundingBoxes);

void uniteMasks(std::vector<cv::Mat>& masks, cv::Mat& dst);

#endif