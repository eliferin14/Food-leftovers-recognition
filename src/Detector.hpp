#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

void featureDetector(Mat& src, vector<KeyPoint>& keypoints, Mat& descriptors);
void clusterKeyPoints(Mat& src);
Mat Postprocess(Mat input_image);
void ignoreIsolatedKeypoints(Mat& src, vector<KeyPoint>& keypoints, vector<KeyPoint>& filteredKeyPoints);

void drawPath(Mat& img, vector<Point2f>& points, Scalar color);

// Mean Shift for keypoints
double pointsDistance(Point p1, Point p2);
void getNeighbourhood(vector<KeyPoint>& keypoints, Point center, double radius, vector<KeyPoint>& neighbourhood);
Point2f getBaricenter(vector<KeyPoint>& neighbourhood);
void meanShift_onePoint(vector<KeyPoint>& keypoints, Point2f startingPoint, double radius, double threshold, vector<Point2f>& baricenterIterations);
void meanShift_grid(Mat& src, vector<KeyPoint>& keypoints, double radius, double threshold, vector<vector<Point2f>>& paths, int gridRows, int gridCols);
void meanShift_keypoints(Mat& src, vector<KeyPoint>& keypoints, double radius, double threshold, vector<vector<Point2f>>& paths);
void findCentroids(vector<vector<Point2f>>& paths, double radius, vector<Point2f>& centroids);

void removeLowSaturationHSV(Mat& src, Mat& mask, double threshold);