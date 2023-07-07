#include "sliceMatch.h"
#include<cmath>
#include <opencv2/core/types.hpp>
#include"Detector.hpp"
using namespace std;
using namespace cv;

void featurePruning(vector<Point2f>centers, vector<Point2f>points, vector<int>labels, vector<Point2f>&prunedPoints, vector<int>&newLabels, double threshold) {
	for (int i = 0; i < points.size(); i++) {
		for (int j = 0; j < centers.size(); j++) {
			if (labels[i] == j) {
				if (distance(centers[j], points[i]) < threshold) {
					prunedPoints.push_back(points[i]);
					newLabels.push_back(labels[i]);
				}
			}
		}
	}
}

float distance(Point2f a, Point2f b) {
	float dist = 0.0f;
	float dist_y = (a.y - b.y);
	float dist_x = (a.x - b.x);
	dist = dist_y*dist_y + dist_x*dist_x;
	dist = pow(dist,0.5);
	return dist;
}

void kMeans(vector<Point2f> centers, vector<Point2f>points, vector<int> &labels, int nCluster) {
	float temp_dist = 1000;
	float temp = 0;
	for (int i = 0; i < points.size(); i++) {
		for (int j = 0; j < centers.size(); j++) {
			temp = distance(centers[j], points[i]);
			if (temp < temp_dist) {
				labels[i] = j;
				temp_dist = temp;
			}
		}
		temp_dist = 1000;
	}
}

void clusterPruning(vector<Point2f>& centers, vector<int> labels,int threshold) {
	vector<int> counter(centers.size());
	for (int i = 0; i < labels.size(); i++) {
		counter[labels[i]]++;
	}
	vector<Point2f> newCenters;
	for (int i = 0; i < counter.size(); i++) {
		if (counter[i] >= threshold) {
			newCenters.push_back(centers[i]);
		}
	}
	centers = newCenters;
}

void imageSlicer(vector<vector<Point2f>> coordinatePoints,Mat originalImage,vector<Mat>& slicedImages) {
	//We get the coordinatesof the bounding box so that we can slice the image for each bounding box.
	int temp_x1, temp_x2, temp_y1, temp_y2;
	for (int i = 0; i < coordinatePoints.size(); i++) {
		temp_x1 = coordinatePoints[i][0].x;
		temp_y1 = coordinatePoints[i][0].y;
		temp_x2 = coordinatePoints[i][1].x - temp_x1;
		temp_y2 = coordinatePoints[i][1].y - temp_y1;
		slicedImages.push_back(originalImage(Rect(temp_x1,temp_y1,temp_x2,temp_y2)));
	}

}

void slicerViewer(vector<Mat>srcs, vector < vector<KeyPoint>>& keys, vector<Mat>& descriptors) {
	for (int i = 0; i < srcs.size(); i++) {
		vector<KeyPoint> temp;
		Mat temp2;
		keys.push_back(temp);
		descriptors.push_back(temp2);
		featureDetector(srcs[i], keys[i], descriptors[i]);
	}
}

void slicedFeatureViewer(vector<Mat> slicedImages,vector<vector<KeyPoint>>slicedKeypoints) {
	Scalar colorTab[] =
	{
		Scalar(0, 0, 255),
		Scalar(0,255,0),
		Scalar(255,100,100),
		Scalar(255,0,255),
		Scalar(0,255,255)
	};
	vector<vector<int>>slicedLabels(slicedImages.size());
	vector<vector<Point2f>>slicedCenters(slicedImages.size());
	vector<vector<Point2f>>slicedPoints(slicedImages.size());
	TermCriteria criteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 10, 1);
	for (int j = 0; j < slicedImages.size(); j++) {
		for (int i = 0; i < slicedKeypoints[j].size(); i++) {
			//cout << "here" << j<< " "<<i<< endl;
			slicedPoints[j].push_back(slicedKeypoints[j][i].pt);
		}
		kmeans(slicedPoints[j], 1, slicedLabels[j], criteria, 10, KMEANS_RANDOM_CENTERS, slicedCenters[j]);
	}
	//cout << (slicedPoints.size()==slicedImages.size()) << endl;
	for (int j = 0; j < slicedImages.size(); j++) {
		for (int i = 0; i < slicedCenters[j].size(); i++)
		{
			Point2f c = slicedCenters[j][i];
			circle(slicedImages[j], c, 40, colorTab[j], 1, LINE_AA);
		}
		for (int i = 0; i < slicedPoints[j].size(); i++) {
			Point2f c = slicedPoints[j][i];
			circle(slicedImages[j], c, 10, colorTab[j], 1, LINE_AA);
		}
	}
	for (int i = 0; i < slicedImages.size(); i++) {
		namedWindow(to_string(i), WINDOW_NORMAL);
		imshow(to_string(i), slicedImages[i]);
		waitKey(0);
	}
}

void cicleMatcher(Mat target, Mat candidate, vector<int>& score) {
	vector<KeyPoint> srcKeyPoints;
	vector<KeyPoint> trgKeyPoints;
	Mat srcDescriptors;
	Mat trgDescriptors;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	featureDetector(target, trgKeyPoints, trgDescriptors);
	featureDetector(candidate, srcKeyPoints, srcDescriptors);			//This is gonna be changed with a constant array of images

	//OpenCV tutorial
	std::vector< std::vector<DMatch> > knn_matches;
	matcher->knnMatch(trgDescriptors, srcDescriptors, knn_matches, 2);
	const float ratio_thresh = 0.75f;
	std::vector<DMatch> good_matches;
	int numMatches = 0;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			good_matches.push_back(knn_matches[i][0]);
			numMatches++;
		}
	}
	score.push_back(numMatches);
	cout << numMatches << endl;
}
const Mat pastaSugo = imread("../ourDataset/pasta1.jpg");
const Mat insalata = imread("../ourDataset/insalata3.jpg");
const Mat patate = imread("../ourDataset/patate2.jpg");
const Mat carne = imread("../ourDataset/carne2.jpg");
const int dataSize = 5;
vector<Mat> pasta;
vector<Mat> meat;
vector<Mat> salad;
vector<Mat> potatoes;
const int numClasses = 4;//Soon it will be 13 :)
vector<int> scores(numClasses);
vector<int> checks(numClasses);
const vector<String> labels = {"Pasta","Meat","Salad","Potatoes"};
vector<vector<Mat>> allClasses;
void matcherInitializer() {
	for (int i = 0; i < dataSize; i++) {
		pasta.push_back(imread("../ourDataset/pasta" + to_string(i+1) + ".jpg"));
		meat.push_back(imread("../ourDataset/carne" + to_string(i+1) + ".jpg"));
		salad.push_back(imread("../ourDataset/insalata" + to_string(i+1) + ".jpg"));
		potatoes.push_back(imread("../ourDataset/patate" + to_string(i+1) + ".jpg"));
	}
	allClasses.push_back(pasta);
	allClasses.push_back(meat);
	allClasses.push_back(salad);
	allClasses.push_back(potatoes);
}

void matcher(Mat target,vector<int> &probability) {
	matcherInitializer();
	vector<Mat> candidates = { insalata,patate ,pastaSugo,carne};
	for (int i = 0; i < candidates.size(); i++) {
		cicleMatcher(target,candidates[i],probability);
	}
}

void classifier(Mat target) {
	//The number 4 is gonna be changed into a constant soon
	for (int classes = 0; classes < numClasses;classes++) {
		for (int samples = 0; samples < numClasses; samples++) {
			cicleMatcher2(target,allClasses[classes][samples],classes);
		}
	}
	getScores();
	bestScore();
	cleanScores();
}

void getScores() {
	for (int i = 0; i < scores.size(); i++)
		cout << labels[i] << " got a score of " << scores[i] << endl;
}
void cleanScores() {
	for (int i = 0; i < scores.size(); i++) {
		scores[i] = 0;
		/*checks[i] = 0;*/
	}
}

void bestScore() {
	int currentMax = 0;
	int index = 0;
	for (int i = 0; i < scores.size(); i++) {
		if (currentMax < scores[i] && checks[i]!=1) {
			currentMax = scores[i];
			index = i;
		}
	}
	checks[index] = 1;
	cout << "The subject contains: " << labels[index] << endl;
	cout << "And it got a score of: " << currentMax << endl;
}

void cicleMatcher2(Mat target, Mat candidate,int index) {
	vector<KeyPoint> srcKeyPoints;
	vector<KeyPoint> trgKeyPoints;
	Mat srcDescriptors;
	Mat trgDescriptors;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	featureDetector(target, trgKeyPoints, trgDescriptors);
	featureDetector(candidate, srcKeyPoints, srcDescriptors);			//This is gonna be changed with a constant array of images

	//OpenCV tutorial
	std::vector< std::vector<DMatch> > knn_matches;
	matcher->knnMatch(trgDescriptors, srcDescriptors, knn_matches, 2);
	const float ratio_thresh = 0.95f;
	std::vector<DMatch> good_matches;
	int numMatches = 0;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			good_matches.push_back(knn_matches[i][0]);
			numMatches++;
		}
	}
	scores[index] += numMatches;
	//cout << numMatches << endl;
}

void removeLowSaturationHSV(Mat& src, Mat& mask, double threshold) {

	// Assuming the src is passed as BGR image, we convert it to HSV
	Mat srcHSV;
	cvtColor(src, srcHSV, COLOR_BGR2HSV);

	// Select the saturation channel (the second)
	vector<Mat> channels;
	split(srcHSV, channels);
	Mat srcS = channels[1];

	// Remove all the pixel with saturation lower than the threshold and generate the mask
	inRange(srcS, threshold, 255, mask);
}