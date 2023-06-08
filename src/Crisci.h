#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

void featurePruning(vector<Point2f>centers, vector<Point2f>points,vector<int>labels,vector<Point2f>&prunedPoints, vector<int>&newLabels,double threshold);
float distance(Point2f a, Point2f b);
void kMeans(vector<Point2f> centers,vector<Point2f>points,vector<int>& labels,int nCluster);
void clusterPruning(vector<Point2f>& centers, vector<int> labels,int threshold); //We might need the & next to labels due to best practice