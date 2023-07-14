#ifndef UTILS
#define UTILS

#include <opencv2/opencv.hpp>

void showImage(std::string windowName, cv::Mat img);
void drawHistogram(cv::Mat& src, cv::Mat& hist, int hist_w, int hist_h, int bins);
void showHistogram(std::string windowName, cv::Mat& src);

#endif