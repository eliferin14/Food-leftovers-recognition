#include "Crisci.h"
#include<cmath>
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