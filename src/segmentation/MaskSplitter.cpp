#include "MaskSplitter.hpp"

cv::Mat detectedEdges;
cv::Mat maskCopy, postMS, postMSgray;

int spatialRadius=50, colorRadius=50;
int cannyLowThreshold=0, ratio=3;

std::string windowNameMS = "Mean Shift";
std::string windowNameCanny = "Detected Edges";

void meanShift(int, void*) {
    cv::pyrMeanShiftFiltering(maskCopy, postMS, spatialRadius, colorRadius);
    cv::imshow(windowNameMS, postMS);
}

void cannyED(int, void*) {
    cv::Mat blurred;
    cv::GaussianBlur(postMSgray, blurred, cv::Size(5,5), 0);
    cv::Canny(blurred, detectedEdges, cannyLowThreshold, cannyLowThreshold*ratio);
    cv::imshow(windowNameCanny, detectedEdges);
}

void splitMask(cv::Mat& inputMask, std::vector<cv::Mat>& outputMasks) {
    /*
    // Converto to HSV
    cv::Mat maskCopy = inputMask.clone();
    cv::cvtColor(maskCopy, maskCopy, cv::COLOR_BGR2HSV);

    // Mean shift
    cv::Mat postMS;
    double spatialRadius = 50;
    double colorRadius = 100;
    cv::pyrMeanShiftFiltering(maskCopy, postMS, spatialRadius, colorRadius);

    showImage("Post MS", postMS);

    cv::Mat gray;
    cv::cvtColor(postMS, gray, cv::COLOR_HSV2BGR);
    cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);
    showImage("Gray", gray);
    cv::Mat hist;
    drawHistogram(gray, hist, 512, 256, 256);
    showImage("Histogram", hist);

    // Split the masks
    */

    maskCopy = inputMask.clone();
    //cv::pyrDown(inputMask, maskCopy);

    // Mean Shift with trackbar
    cv::namedWindow(windowNameMS);
    cv::createTrackbar("Spatial radius", windowNameMS, &spatialRadius, 100, meanShift);
    cv::createTrackbar("Color radius", windowNameMS, &colorRadius, 100, meanShift);

    meanShift(0,0);

    cv::waitKey(0);

    // Canny edge detector
    cv::namedWindow(windowNameCanny);
    cv::cvtColor(postMS, postMSgray, cv::COLOR_BGR2GRAY);
    cv::createTrackbar("Canny low threshold", windowNameCanny, &cannyLowThreshold, 100, cannyED);

    cannyED(0,0);

    cv::waitKey(0);

    // Contours
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat contoursImage = inputMask.clone();
    cv::findContours(detectedEdges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(contoursImage, contours, -1, cv::Scalar(0,0,255));
    cv::namedWindow("Contours");
    cv::imshow("Contours", contoursImage);
    cv::waitKey(0);

}