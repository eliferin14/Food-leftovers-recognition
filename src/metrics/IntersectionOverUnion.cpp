#include "IntersectionOverUnion.hpp"

int countPixelOfMask(cv::Mat& mask) {
    int counter = 0;
    for (int r=0; r<mask.rows; r++) {
        for (int c=0; c<mask.cols; c++) {
            if (mask.at<u_int8_t>(r,c) > 0 ) {
                counter++;
            }
        }
    }
    return counter;
}

double iou_twoMasks(cv::Mat& mask, cv::Mat& trueMask) {

    if (mask.size() != trueMask.size()) {
        std::cerr << "Incorrect mask size" << std::endl;
        exit(-1);
    }

    cv::inRange(trueMask, 1, 255, trueMask);

    cv::Mat intersection, union_;

    cv::bitwise_and(mask, trueMask, intersection);
    cv::bitwise_or(mask, trueMask, union_);

    int iCounter = countPixelOfMask(intersection);
    int uCounter = countPixelOfMask(union_);

    double iou = (double)iCounter / uCounter;

    if (iou > 1 || iou < 0) {
        std::cerr << "IoU impossible value" << std::endl;
        exit(-1);
    }

    return iou;
}

void getUnionMaskFromBB(std::vector<cv::Rect>& boundingBoxes, cv::Mat& mask) {
    // We assume mask to be a zero matrix with already the correct dimensions

    // Draw the rectangles
    for (int i=0; i<boundingBoxes.size(); i++) {
        cv::rectangle(mask, boundingBoxes[i], cv::Scalar(255), -1, 8, 0);
    }
}

double iou_twoImagesUnionBB(std::vector<cv::Rect>& bb, std::vector<cv::Rect>& trueBB) {
    cv::Mat mask = cv::Mat::zeros(cv::Size(1280, 960), CV_8UC1);
    cv::Mat trueMask = mask.clone();

    getUnionMaskFromBB(bb, mask);
    getUnionMaskFromBB(trueBB, trueMask);

    return iou_twoMasks(mask, trueMask);
}