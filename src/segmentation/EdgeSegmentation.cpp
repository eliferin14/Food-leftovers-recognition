#include "EdgeSegmentation.hpp"

void edgeDetector(Mat& src, Mat& edges) {
    // NB: could get better result by applying the edge detector on some specific channel
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    // Blur?
    GaussianBlur(gray, gray, Size(3,3), 0);

    // Canny edge detector
    int lowThreshold = 30;
    int ratio = 3;
    int kernelSize = 3;
    Canny(gray, edges, lowThreshold, lowThreshold*ratio, kernelSize);

    // Laplacian
    Laplacian(gray, edges, CV_16S);
    convertScaleAbs(edges, edges);
}

void edgeDensity(Mat& edges, Mat& edgeDensity, int windowSize) {
    Mat kernel = Mat(Size(windowSize, windowSize), CV_32FC1, Scalar(1)) / (windowSize*windowSize);

    filter2D(edges, edgeDensity, 1, kernel);
}

double computeKPDensity(int x, int y, vector<KeyPoint>& keypoints, double radius) {
    double density = 0;
    Point center(x, y);

    for ( int i=0; i<keypoints.size(); i++ ) {
        double distance = pointsDistance( center, keypoints[i].pt );
        if ( distance <= radius ) {
            density += ( radius - distance ) * keypoints[i].size;
        }
    }

    return density;
}

void keyPointDensity(Mat& src, Mat& density, double radius) {
    // Detect keypoints
    vector<KeyPoint> keypoints;
    Mat descriptors;
    featureDetector(src, keypoints, descriptors);

    // For each pixel compute density
    density = Mat(src.size(), CV_64FC1, Scalar(0));
    double maxDensity = 0;
    for (int x=0; x<density.cols; x++) {
        for (int y=0; y<density.rows; y++) {
            double pixelDensity = computeKPDensity(x,y, keypoints, radius);
            density.at<double>(x,y) = pixelDensity;
            if (pixelDensity > maxDensity) maxDensity = pixelDensity;
        }
    }

    for (int x=0; x<density.cols; x++) {
        for (int y=0; y<density.rows; y++) {
            density.at<double>(x,y) /= maxDensity;
        }
    }
    
}