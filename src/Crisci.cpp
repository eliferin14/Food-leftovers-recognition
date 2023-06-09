#include "Crisci.h"
#include<cmath>
#include <opencv2/core/types.hpp>
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