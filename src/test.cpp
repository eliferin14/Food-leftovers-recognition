#include <iostream>
#include "opencv2/opencv.hpp"
#include "Detector.hpp"

using namespace cv;
using namespace std;

void showImage(string windowName, Mat img) {
    namedWindow(windowName);
    imshow(windowName, img);
    waitKey(0);
}

void test_pointsDistance() {
    Point2f p1(-3,-4);
    Point2f p2(3,4);
    double distance = pointsDistance(p1,p2);
    cout << distance << endl;
}

void test_getNeighbourhood() {
    Mat image = imread("../Food_leftover_dataset/tray4/food_image.jpg");

    vector<KeyPoint> keypoints;
    Mat descriptors;
    featureDetector(image, keypoints, descriptors);

    Point2f center(450, 50);
    double radius = 50;
    vector<KeyPoint> neighbourhood;
    getNeighbourhood(keypoints, center, radius, neighbourhood);
    Mat neighbourPoints;
    drawKeypoints(image, neighbourhood, neighbourPoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    circle(neighbourPoints, center, radius, (255,0,0), 2, 8, 0);
    showImage("Neighbourhood", neighbourPoints);
}

void test_getBaricenter() {
    Mat image = imread("../Food_leftover_dataset/tray4/food_image.jpg");

    vector<KeyPoint> keypoints;
    Mat descriptors;
    featureDetector(image, keypoints, descriptors);

    Point2f center(450, 150);
    double radius = 50;
    vector<KeyPoint> neighbourhood;
    getNeighbourhood(keypoints, center, radius, neighbourhood);
    Mat neighbourPoints;
    drawKeypoints(image, neighbourhood, neighbourPoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    circle(neighbourPoints, center, radius, (255,0,0), 2, 8, 0);

    Point2f baricenter = getBaricenter(neighbourhood);
    circle(neighbourPoints, baricenter, 5, (0,255,0), 2, 8, 0);
    showImage("Neighbourhood", neighbourPoints);
}

void test_meanShift_onePoint() {
    Mat image = imread("../Food_leftover_dataset/tray4/food_image.jpg");

    vector<KeyPoint> keypoints;
    Mat descriptors;
    featureDetector(image, keypoints, descriptors);

    Point2f center(830, 500);
    double radius = 150;
    double threshold = 0.5;
    vector<Point2f> iter;
    meanShift_onePoint(keypoints, center, radius, threshold, iter);

    cout << iter << endl;

    Mat path = image.clone();
    drawPath(path, iter, Scalar(255,0,0));
    showImage("Path", path);
}

void test_meanShift_grid() {
    Mat image = imread("../Food_leftover_dataset/tray8/food_image.jpg");

    vector<KeyPoint> keypoints;
    Mat descriptors;
    featureDetector(image, keypoints, descriptors);

    double radius = 150;
    double threshold = 0.5;
    int gridRows = 10;
    int gridCols = 10;
    vector<vector<Point2f>> paths;

    meanShift_grid(image, keypoints, radius, threshold, paths, gridCols, gridRows);

    for (int i=0; i<paths.size(); i++) {
        drawPath(image, paths[i], Scalar(255,0,0));
    }
    showImage("Mean shift grid", image);
}

int main(int argc, char** argv) {
    
    test_meanShift_grid();

}