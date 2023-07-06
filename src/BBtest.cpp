#include <iostream>
#include <opencv2/opencv.hpp>
#include "BoundBox.hpp"
#include "Detector.hpp"
#include "Crisci.hpp"

using namespace std;
using namespace cv;

void showImage(string windowName, Mat img) {
    namedWindow(windowName);
    imshow(windowName, img);
    waitKey(0);
}

int main(int argc, char** argv) {

    char* filename = argv[1];

    Mat image = imread(filename);
    showImage("Original", image);

    // Extract the keypoints and find the centroids using meanshift
    vector<KeyPoint> keypoints;
    Mat descriptors;
    featureDetector(image, keypoints, descriptors);

    double radius = 200;
    double threshold = 0.5;
    vector<vector<Point2f>> paths;

    meanShift_keypoints(image, keypoints, radius, threshold, paths);

    double centroidRadius = 200;
    vector<Point2f> centroids;
    findCentroids(paths, centroidRadius, centroids);

    // Prune the keypoints in order to define the bounding boxes
    Scalar colorTab[] =
    {
        Scalar(0, 0, 255),
        Scalar(0,255,0),
        Scalar(255,0,255),
        Scalar(255,255,0),
        Scalar(0,255,255)
    };
    Mat cleanImage = image.clone();
    vector<int> labels(keypoints.size());
    vector<Point2f>points;
    for (int i = 0; i < keypoints.size(); i++) {
            points.push_back(keypoints[i].pt);
        }
    kMeans(centroids,points,labels,centroids.size());
    vector<Point2f> prunedPoints;
    vector<int> newLabels;
    double threshold2 = 325;
    int threshold3 = 100;
    clusterPruning(centroids, labels, threshold3);
    kMeans(centroids, points, labels, centroids.size());
    featurePruning(centroids, points, labels, prunedPoints, newLabels, threshold2);
    Mat finalImage = cleanImage.clone();
    for (int i = 0; i < prunedPoints.size(); ++i)
    {
        Point2f c = prunedPoints[i];
        circle(finalImage, c, 10, colorTab[newLabels[i]], 1, LINE_AA);
    }
    showImage("Clusters", finalImage);


    //Crea le Bounding Box
    vector<vector<Point2f>> extremes;
    BoundingBoxID(finalImage,prunedPoints,centroids,newLabels, extremes);
    
    //Now let's divide the images
    vector<Mat> slicedImages;
    imageSlicer(extremes, cleanImage , slicedImages);

    // For each bounding box, we run again the mean shift on the keypoints and keep only the keypoints of the largest cluster
    // => All the outliers of the keypoint vector are ignored

    vector<vector<Point2f>> newCorners(slicedImages.size());
    for (int i=0; i<slicedImages.size(); i++) {
        refineBoundingBox(slicedImages[i], extremes[i], newCorners[i], 100, 0.5, 250, false);
    }

    Mat refinedBB = image.clone();
    drawBoundingBoxes(refinedBB, newCorners);
    showImage("refinedBB", refinedBB);

}