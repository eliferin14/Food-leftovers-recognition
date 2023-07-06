#include "Detector.hpp"

using namespace std;
using namespace cv;


void featureDetector(Mat& src, vector<KeyPoint>& keypoints, Mat& descriptors) {

    // SIFT parameters
    // https://docs.opencv.org/3.4/d7/d60/classcv_1_1SIFT.html#ad337517bfdc068ae0ba0924ff1661131
    int nFeatures = 0;                  // 0
    int nOctaveLayers = 3;              // 3
    double contrastThreshold = 0.05;    // 0.04
    double edgeThreshold = 5;          // 10
    double sigma = 1.6;                 // 1.6

    // Feature detector implementation, can be SIFT, SURF, ORB...
    Ptr<SIFT> detectorPtr = SIFT::create(nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    //Ptr<xfeatures2d::SURF> detectorPtr = xfeatures2d::SURF::create();
    //Ptr<ORB> detectorPtr = ORB::create();

    detectorPtr->detect(src, keypoints);
    detectorPtr->compute(src, keypoints, descriptors);

    bool flag = 1;
    if (flag) {
        Mat src_keypoints;
        drawKeypoints(src, keypoints, src_keypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        /*namedWindow("Keypoints");
        imshow("Keypoints", src_keypoints);
        waitKey(0);*/
    }
}

double pointsDistance(Point p1, Point p2) {
    float deltaX = p1.x - p2.x;
    float deltaY = p1.y - p2.y;

    float distance = sqrt(pow(deltaX, 2) + pow(deltaY, 2));

    return distance;
}

void getNeighbourhood(vector<KeyPoint>& keypoints, Point center, double radius, vector<KeyPoint>& neighbourhood) {
    // Scan all the keypoints and select the ones with distance from the center < radius
    for ( int i=0; i<keypoints.size(); i++ ) {
        if ( pointsDistance( center, keypoints[i].pt ) <= radius ) {
            neighbourhood.push_back(keypoints[i]);
        }
    }
}

Point2f getBaricenter(vector<KeyPoint>& neighbourhood) {
    // If there are no neighbours, return (-1,-1);
    if (neighbourhood.size()==0) {
        return Point2f(-1,-1);
    }

    // Compute the mean x and y of all the points
    float sumX=0, sumY=0;
    for (int i=0; i<neighbourhood.size(); i++) {
        sumX += neighbourhood[i].pt.x;
        sumY += neighbourhood[i].pt.y;
    }

    double barX = sumX / neighbourhood.size();
    double barY = sumY / neighbourhood.size();

    Point2f baricenter(barX, barY);

    return baricenter;
}

void meanShift_onePoint(vector<KeyPoint>& keypoints, Point2f startingPoint, double radius, double threshold, vector<Point2f>& baricenterIterations) {
    Point2f center = startingPoint;
    baricenterIterations.push_back(center);
    double baricenterDisplacement = radius;
    
    // Iterate until the displacement between iterations is larger than the given threshold
    while (baricenterDisplacement >= threshold) {
        // Find the keypoints in the neighbourhood
        vector<KeyPoint> neighbours;
        getNeighbourhood(keypoints, center, radius, neighbours);

        // Update the baricenter
        Point2f oldCenter = center;
        center = getBaricenter(neighbours);

        // If there are no neighbours (baricenter is (-1,-1)), end the cycle
        if (center.x == -1 && center.y == -1) {
            return;
        }

        baricenterIterations.push_back(center);

        // Compute the displacement
        baricenterDisplacement = pointsDistance(center, oldCenter);
    }
}

void drawPath(Mat& img, vector<Point2f>& points, Scalar color) {
    if ( points.size() == 1 ) {
        return;
    }

    // Draw the starting point in green
    circle(img, points.front(), 3, Scalar(0,255,0), 3, 8, 0);

    // Blue arrows between iterations
    for (int i=0; i<points.size()-1; i++) {
        arrowedLine(img, points[i], points[i+1], color, 2, 8, 0, 0.1);
    }

    // Draw the end point in red
    circle(img, points.back(), 3, Scalar(0,0,255), 3, 8, 0);
}

void meanShift_grid(Mat& src, vector<KeyPoint>& keypoints, double radius, double threshold, vector<vector<Point2f>>& paths, int gridRows, int gridCols) {
    // Generate the grid points
    vector<Point2f> grid;
    double verticalDelta = src.rows / gridRows;
    double horizontalDelta = src.cols / gridCols;

    for (int r=0; r<gridRows; r++) {
        for (int c=0; c<gridCols; c++) {
            double x = 0.5*horizontalDelta + horizontalDelta*c;
            double y = 0.5*verticalDelta + verticalDelta*r;
            Point2f p(x, y);
            grid.push_back(p);
        }
    }

    // For each point in the grid, execute the meanShift algorithm and save the path
    for (int i=0; i<grid.size(); i++) {
        vector<Point2f> path;
        meanShift_onePoint(keypoints, grid[i], radius, threshold, path);
        paths.push_back(path);
    }
}

void meanShift_keypoints(Mat& src, vector<KeyPoint>& keypoints, double radius, double threshold, vector<vector<Point2f>>& paths) {
    for (int i=0; i<keypoints.size(); i++) {
        vector<Point2f> path;
        meanShift_onePoint(keypoints, keypoints[i].pt, radius, threshold, path);
        paths.push_back(path);
    }
}

void findCentroids(vector<vector<Point2f>>& paths, double radius, vector<Point2f>& centroids) {
    // Scan all the paths
    for (int i=0; i<paths.size(); i++) {

        // If the path has only 1 point, ignore it
        if (paths[i].size()==1) continue;

        // Otherwise we select the final point
        Point2f finalPoint = paths[i].back();

        // We check if the final point is close to a point that is already a centroid
        // If there are no close centroids, we include it in the vector. Otherwise we ignore it
        bool isClose = false;
        for (int j=0; j<centroids.size(); j++) {
            if (pointsDistance(centroids[j], finalPoint) < radius) {
                isClose = true;

                // We update the path by adding the centroid as last element
                paths[i].push_back(centroids[j]);

                break;
            }
        }
        if ( !isClose ) centroids.push_back(finalPoint);

    }
}