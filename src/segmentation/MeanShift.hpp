#ifndef MEAN_SHIFT
#define MEAN_SHIFT

#include <opencv2/opencv.hpp>

// SIFT feature detector
void featureDetector(cv::Mat& src, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors); //Returns a std::vector of keypoints

// Mean shift helpers
double pointsDistance(cv::Point p1, cv::Point p2);  // Compute the geometric distance between keypoints
void getNeighbourhood(std::vector<cv::KeyPoint>& keypoints, cv::Point center, double radius, std::vector<cv::KeyPoint>& neighbourhood);   // Given a center point and a std::vector of keypoints, returns all the points within a given radius
cv::Point2f getBaricenter(std::vector<cv::KeyPoint>& neighbourhood); // Return the geometric baricenter of a set of points

// Mean shift implementation
void meanShift_onePoint(std::vector<cv::KeyPoint>& keypoints, cv::Point2f startingPoint, double radius, double threshold, std::vector<cv::Point2f>& baricenterIterations);    // Compute the mean shift path of a single starting point
void meanShift_keypoints(cv::Mat& src, std::vector<cv::KeyPoint>& keypoints, double radius, double threshold, std::vector<std::vector<cv::Point2f>>& paths);   // Iterates the _onecv::Point for the whole std::vector of keypoints
void findCentroids(std::vector<std::vector<cv::Point2f>>& paths, double radius, std::vector<cv::Point2f>& centroids);  // Define the convergence points of the paths

void drawPath(cv::Mat& img, std::vector<cv::Point2f>& points); // Draw the paths on the image

// Clustering
void assignLabels(std::vector<cv::Point2f>& centroids, std::vector<std::vector<cv::Point2f>>& paths, std::vector<int>& labels, std::vector<std::vector<cv::Point2f>>& clusters);
void removeLowCountClusters(std::vector<std::vector<cv::Point2f>>& clusters, std::vector<cv::Point2f>& centroids, int threshold);

// Gaussian pruning
void computeMean(std::vector<std::vector<cv::Point2f>>& clusters, std::vector<cv::Point2f>& means);
double computeCovarianceScalar(std::vector<double> valuesA, std::vector<double> valuesB, double meanA, double meanB);
void computeVarianceMatrices(std::vector<std::vector<cv::Point2f>>& clusters, std::vector<cv::Point2f>& means, std::vector<cv::Mat>& covMatrices);
cv::RotatedRect getErrorEllipse(double chisquare_val, cv::Point2f mean, cv::Mat covmat);
void gaussianPruning(std::vector<std::vector<cv::Point2f>>& clusters, std::vector<cv::Point2f>& means, std::vector<cv::Mat>& covMatrices, double varianceThreshold, std::vector<cv::RotatedRect>& ellipses);

// Gaussian clustering
double gaussianLikelihood(cv::Point2f p, cv::Point2f mean, cv::Mat covmat);
void gaussianClustering(std::vector<cv::KeyPoint>& keypoints, std::vector<cv::Point2f>& means, std::vector<cv::Mat>& covMatrices, std::vector<std::vector<cv::Point2f>>& clusters);

// Distance pruning
void distancePruning(std::vector<std::vector<cv::Point2f>>& clusters, std::vector<cv::Point2f>& centroids, double distanceThreshold);

// Kmeans
void kmeansClustering(std::vector<cv::KeyPoint>& keypoints, std::vector<cv::Point2f>& centroids, std::vector<std::vector<cv::Point2f>>& clusters);

#endif