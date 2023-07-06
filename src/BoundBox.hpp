#include "opencv2/opencv.hpp"
#include "Detector.hpp"

using namespace std;
using namespace cv;


void BoundingBoxID(Mat& src, std::vector<Point2f> points, std::vector<Point2f> centers, vector<int> bestLabels, vector<vector<Point2f>>& extreme);
void refineBoundingBox(Mat& src, vector<Point2f>& inputCorners, vector<Point2f>& outputCorners, double MSradius, double threshold, double centroidRadius, bool showResult=false);
void drawBoundingBoxes(Mat& img, vector<vector<Point2f>>& corners);
