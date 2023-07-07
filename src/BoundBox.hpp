#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;


void BoundingBoxID(Mat& src, std::vector<Point2f> points, std::vector<Point2f> centers, vector<int> bestLabels, vector<vector<Point2f>>& extremes);

void RefinedBoxId(Mat& src, vector<vector<Point2f>>& extremes, Point2f centers);