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

void classifier(Mat target,Point2f center, String& composition) {
	Mat cloned[3];
	//composition += "The plate contains: ";
	split(target, cloned);
	//I'll use just the second channel since it is the one who brings the most information
	if ((meanCalculator(cloned[2], 9,center) >= 195 && meanCalculator(cloned[2], 9,center) <= 196)
		|| (meanCalculator(cloned[2], 9, center) >= 79 && meanCalculator(cloned[2], 9, center) <= 80) 
		|| (meanCalculator(cloned[2], 9, center) >= 126 && meanCalculator(cloned[2], 9, center) <= 126.5)) {
		namedWindow("bread", WINDOW_NORMAL);
		imshow("bread", target);
		//cout << "it's bread " << endl;
		cout << " value " << meanCalculator(cloned[2], 9, center) << endl;
		composition += "Bread, ";
		//cout << "channel " << i << " value " << int(cloned[i].at<uchar>(int(cloned[i].rows / 2), int(cloned[i].cols / 2))) << endl;
	}
	else if ((meanCalculator(cloned[2], 9, center) >= 130 && meanCalculator(cloned[2], 9, center) <= 131)) {
		namedWindow("Pasta Pesto", WINDOW_NORMAL);
		imshow("Pasta Pesto", target);
		//cout << "it's Pasta with Pesto " << endl;
		cout << " value " << meanCalculator(cloned[2], 9, center) << endl;
		composition += "Pasta Pesto, ";
		//cout << "channel " << i << " value " << int(cloned[i].at<uchar>(int(cloned[i].rows / 2), int(cloned[i].cols / 2))) << endl;
	}
	else if ((meanCalculator(cloned[2], 9, center) >= 82 && meanCalculator(cloned[2], 9, center) <= 83)
		|| (meanCalculator(cloned[2], 9, center) >= 99 && meanCalculator(cloned[2], 9, center) <= 100)) {
		namedWindow("Pasta Tomato", WINDOW_NORMAL);
		imshow("Pasta Tomato", target);
		//cout << "it's Pasta with Tomato Sauce " << endl;
		cout << " value " << meanCalculator(cloned[2], 9, center) << endl;
		composition += "Pasta Tomato, ";
		//cout << "channel " << i << " value " << int(cloned[i].at<uchar>(int(cloned[i].rows / 2), int(cloned[i].cols / 2))) << endl;
	}
	else if ((meanCalculator(cloned[2], 9, center) >= 118 && meanCalculator(cloned[2], 9, center) <= 119)) {
		namedWindow("Pasta Meat", WINDOW_NORMAL);
		imshow("Pasta Meat", target);
		//cout << "it's Pasta with Meat Sauce " << endl;
		composition += "Pasta Meat, ";
		cout << " value " << meanCalculator(cloned[2], 9, center) << endl;
		//cout << "channel " << i << " value " << int(cloned[i].at<uchar>(int(cloned[i].rows / 2), int(cloned[i].cols / 2))) << endl;
	}
	else if ((meanCalculator(cloned[2], 9, center) >= 75 && meanCalculator(cloned[2], 9, center) <= 76)
		|| (meanCalculator(cloned[2], 9, center) >= 86 && meanCalculator(cloned[2], 9, center) <= 87)
		|| (meanCalculator(cloned[2], 9, center) >= 122 && meanCalculator(cloned[2], 9, center) <= 123)) {
		namedWindow("Pasta with clums", WINDOW_NORMAL);
		imshow("Pasta with clums", target);
		//cout << "it's Pasta with clums " << endl;
		composition += "Pasta with clums, ";
		cout << " value " << meanCalculator(cloned[2], 9, center) << endl;
		//cout << "channel " << i << " value " << int(cloned[i].at<uchar>(int(cloned[i].rows / 2), int(cloned[i].cols / 2))) << endl;
	}
	else if ((meanCalculator(cloned[2], 9, center) >= 141 && meanCalculator(cloned[2], 9, center) <= 142)) {
		namedWindow("Rice", WINDOW_NORMAL);
		imshow("Rice", target);
		//cout << "it's Rice " << endl;
		composition += "Rice, ";
		cout << " value " << meanCalculator(cloned[2], 9, center) << endl;
		//cout << "channel " << i << " value " << int(cloned[i].at<uchar>(int(cloned[i].rows / 2), int(cloned[i].cols / 2))) << endl;
	}
	else if ((meanCalculator(cloned[2], 9, center) >= 126.5 && meanCalculator(cloned[2], 9, center) <= 127)
		|| (meanCalculator(cloned[2], 9, center) >= 101 && meanCalculator(cloned[2], 9, center) <= 102)
		|| (meanCalculator(cloned[2], 9, center) >= 121 && meanCalculator(cloned[2], 9, center) <= 122)
		|| (meanCalculator(cloned[2], 9, center) >= 86 && meanCalculator(cloned[2], 9, center) <= 87)
		|| (meanCalculator(cloned[2], 9, center) >= 46 && meanCalculator(cloned[2], 9, center) <= 47)
		|| (meanCalculator(cloned[2], 9, center) >= 107 && meanCalculator(cloned[2], 9, center) <= 108)) {
		namedWindow("Salad", WINDOW_NORMAL);
		imshow("Salad", target);
		//cout << "it's Salad " << endl;
		composition += "Salad, ";
		cout << " value " << meanCalculator(cloned[2], 9, center) << endl;
		//cout << "channel " << i << " value " << int(cloned[i].at<uchar>(int(cloned[i].rows / 2), int(cloned[i].cols / 2))) << endl;
	}
	else {
		namedWindow("something", WINDOW_NORMAL);
		imshow("something", target);
		composition += "Something, ";
		//cout << "it's something " << endl;
		cout << " value " << meanCalculator(cloned[2],9,center) << endl;
	}
}

float meanCalculator(Mat target,int kSize,Point2f center) {
	float average=0;
	//if(center.x > target.cols || center.y > target.rows){}
	//else {
		for (int i = -kSize / 2; i < kSize / 2; i++) {
			for (int j = -kSize / 2; j < kSize / 2; j++) {
				average += target.at<uchar>(int(target.rows/2) + i, int(target.cols/2) + i);
			}
		}
	//}
	return average/(kSize*kSize);
}


//const int dataSize = 5;
//vector<Mat> pasta;
//vector<Mat> meat;
//vector<Mat> salad;
//vector<Mat> potatoes;
//vector<Mat> bread;
//const int numClasses = 5;//Soon it will be 13 :)
//vector<float> scores(numClasses);
//vector<int> checks(numClasses);
//const vector<String> labels = {"Pasta","Meat","Salad","Potatoes","Bread"};
//vector<vector<Mat>> allClasses;
//void matcherInitializer() {
//	for (int i = 0; i < dataSize; i++) {
//		pasta.push_back(imread("../ourDataset/pasta" + to_string(i+1) + ".jpg"));
//		meat.push_back(imread("../ourDataset/carne" + to_string(i+1) + ".jpg"));
//		salad.push_back(imread("../ourDataset/insalata" + to_string(i+1) + ".jpg"));
//		potatoes.push_back(imread("../ourDataset/patate" + to_string(i+1) + ".jpg"));
//		bread.push_back(imread("../ourDataset/pane" + to_string(i+1) + ".jpg"));
//	}
//	allClasses.push_back(pasta);
//	allClasses.push_back(meat);
//	allClasses.push_back(salad);
//	allClasses.push_back(potatoes);
//	allClasses.push_back(bread);
//}

//void classifier(Mat target, vector<float>& personalScores) {
//	//The number 4 is gonna be changed into a constant soon
//	for (int classes = 0; classes < numClasses;classes++) {
//		for (int samples = 0; samples < numClasses; samples++) {
//			cicleMatcher(target,allClasses[classes][samples],classes);
//		}
//	}
//	//int area = target.rows*target.cols*0.01;
//	normalizeScores();
//	for (int i = 0; i < scores.size(); i++)
//		personalScores.push_back(scores[i]);
//	getScores();
//	bestScore();
//	cleanScores();
//}
//
//void findBestLabel(vector<vector<float>> allScores,int num) {
//	vector<float> bestLabels(num);
//	vector<int> classScores(num);
//	vector<int> imageItGotFound(num);
//	for (int i = 0; i < allScores[0].size(); i++) {
//		for (int j = 0; j < num; j++) {
//			if (bestLabels[i] < allScores[j][i]) {
//				bestLabels[i] = allScores[j][i];
//				classScores[i] = i;
//				imageItGotFound[i] = j;
//			}
//		}
//	}
//	for (int i = 0; i < num; i++) {
//		cout << "Best label for class " << classScores[i] << " is " << bestLabels[i] << " leading to label " << labels[classScores[i]] <<endl;
//		cout << "It got found in image " << imageItGotFound[i]<<endl;
//	}
//}
//
//void getScores() {
//	for (int i = 0; i < scores.size(); i++)
//		cout << labels[i] << " got a score of " << scores[i] << endl;
//}
//void cleanScores() {
//	for (int i = 0; i < scores.size(); i++) {
//		scores[i] = 0;
//	}
//}
//
//void normalizeScores() {
//	for (int i = 0; i < scores.size(); i++)
//		scores[i] = scores[i] / dataSize;
//}
//
//void bestScore() {
//	float currentMax = 0;
//	int index = 0;
//	for (int i = 0; i < scores.size(); i++) {
//		if (currentMax < scores[i] && checks[i]!=1) {
//			currentMax = scores[i];
//			index = i;
//		}
//	}
//	checks[index] = 1;
//	cout << "The subject contains: " << labels[index] << endl;
//	cout << "And it got a score of: " << currentMax << endl;
//}
//
//void cicleMatcher(Mat target, Mat candidate,int index) {
//	vector<KeyPoint> srcKeyPoints;
//	vector<KeyPoint> trgKeyPoints;
//	Mat srcDescriptors;
//	Mat trgDescriptors;
//	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_L1);
//	featureDetector(target, trgKeyPoints, trgDescriptors);
//	featureDetector(candidate, srcKeyPoints, srcDescriptors);			//This is gonna be changed with a constant array of images
//
//	//OpenCV tutorial
//	std::vector< std::vector<DMatch> > knn_matches;
//	matcher->knnMatch(trgDescriptors, srcDescriptors, knn_matches, 20);
//	const float ratio_thresh = 0.80f;
//	std::vector<DMatch> good_matches;
//	float numMatches = 0;
//	for (size_t i = 0; i < knn_matches.size(); i++)
//	{
//		if (knn_matches[i][0].distance <= ratio_thresh * knn_matches[i][1].distance)
//		{
//			good_matches.push_back(knn_matches[i][0]);
//			numMatches++;
//		}
//	}
//	scores[index] += numMatches;//trgKeyPoints.size();
//	//cout << numMatches << endl;
//}