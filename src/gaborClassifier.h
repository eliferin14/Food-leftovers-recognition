#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

Mat getGaborKernel(double sigma, double theta, double lambda, double gamma);
int classifier(const Mat& filteredImage);