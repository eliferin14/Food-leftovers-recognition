#include <iostream>
#include <opencv2/opencv.hpp>
#include "../Detector.hpp"

using namespace std;
using namespace cv;

void edgeDetector(Mat& src, Mat& edges);
void edgeDensity(Mat& edges, Mat& edgeDensity, int windowSize);
void edgeDensitySegmentation(Mat& edgeDensity);

void keyPointDensity(Mat& src, Mat& density, double radius);

