#include "BoundingBoxes.hpp"

void getBoundingBoxes(std::vector<std::vector<cv::Point2f>>& clusters, std::vector<cv::Rect>& boundingBoxes) {
    // For each cluster find the extremes and build a rectangle accordingly
    for (int i=0; i<clusters.size(); i++) {

        double minX = 100000;
        double minY = 100000;
        double maxX = -1;
        double maxY = -1;

        for (int j=0; j<clusters[i].size(); j++) {
            double pX = clusters[i][j].x;
            double pY = clusters[i][j].y;

            if ( pX < minX ) minX = pX;
            if ( pY < minY ) minY = pY;
            if ( pX > maxX ) maxX = pX;
            if ( pY > maxY ) maxY = pY;
        }

        // Define the two extremes of the bounding box
        cv::Point2f upperLeftCorner(minX, minY);
        cv::Point2f lowerRightCorner(maxX, maxY);

        // Define the boundingBox and save it in the output vector
        cv::Rect boundingBox(upperLeftCorner, lowerRightCorner);
        boundingBoxes.push_back(boundingBox);
    }
}

void extractPlatesBB(cv::Mat& src, std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& plates) {
    for (int i=0; i<boundingBoxes.size(); i++) {
        // Deep copy of the bounding box in a new image
        cv::Mat subMatrix = src( boundingBoxes[i] ).clone();
        plates.push_back(subMatrix);
    }
}
