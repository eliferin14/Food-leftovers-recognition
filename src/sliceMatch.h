#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

void featurePruning(vector<Point2f>centers, vector<Point2f>points,vector<int>labels,vector<Point2f>&prunedPoints, vector<int>&newLabels,double threshold);
float distance(Point2f a, Point2f b);
void kMeans(vector<Point2f> centers,vector<Point2f>points,vector<int>& labels,int nCluster);
void clusterPruning(vector<Point2f>& centers, vector<int> labels,int threshold); //We might need the & next to labels due to best practice
void imageSlicer(vector<vector<Point2f>> coordinatePoints,Mat originalImage ,vector<Mat>& slicedImages);
void slicerViewer(vector<Mat>srcs,vector<vector<KeyPoint>>& keys,vector<Mat>& descriptors);
void slicedFeatureViewer(vector<Mat> slicedImages, vector<vector<KeyPoint>>slicedKeypoints);

void classifier(Mat target,Point2f center,String& composition);
float meanCalculator(Mat target,int kSize,Point2f center);

void slideClassifier(Mat& target,int windowSize,bool flag);
Mat lookupTable(int levels);
Mat colorReduce(const cv::Mat& image, int levels);
//void slideCounter(Mat target, int windowSize);


//void matcher(Mat target, vector<int>& probability);
//void matcherInitializer();
//void classifier(Mat target,vector<float>& personalScores);
//void bestScore();
//void cleanScores();
//void cicleMatcher(Mat target, Mat candidate, int index);
//void getScores();
//void normalizeScores();
//void findBestLabel(vector<vector<float>> allScores,int numImages);