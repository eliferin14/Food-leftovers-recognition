#include "Detector.hpp"

using namespace std;
using namespace cv;


void featureDetector(Mat& src, vector<KeyPoint>& keypoints, Mat& descriptors) {
    // Feature detector implementation, can be SIFT, SURF, ORB...
    Ptr<SIFT> detectorPtr = SIFT::create();
    //Ptr<xfeatures2d::SURF> detectorPtr = xfeatures2d::SURF::create();
    //Ptr<ORB> detectorPtr = ORB::create();

    detectorPtr->detect(src, keypoints);
    detectorPtr->compute(src, keypoints, descriptors);

    bool flag = 1;
    if (flag) {
        Mat src_keypoints;
        drawKeypoints(src, keypoints, src_keypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        namedWindow("Keypoints");
        imshow("Keypoints", src_keypoints);
        waitKey(0);
    }
}