#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

void featurePruning(vector<Point2f>centers, vector<Point2f>points,vector<int>labels,vector<Point2f>&prunedPoints, vector<int>&newLabels,double threshold);
float distance(Point2f a, Point2f b);