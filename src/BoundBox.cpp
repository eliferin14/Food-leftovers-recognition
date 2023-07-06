#include "BoundBox.hpp"

using namespace std;
using namespace cv; 

void BoundingBoxID(Mat& src, std::vector<Point2f> points, std::vector<Point2f> centers, vector<int> bestLabels, vector<vector<Point2f>>& extremes){

    Mat img=src.clone();

    Scalar colorTab[] =
    {
        Scalar(0, 0, 255),
        Scalar(0,255,0),
        Scalar(255,100,100),
        Scalar(255,0,255),
        Scalar(0,255,255)
    };
    /*
    for(int i=0;i<bestLabels.size();i++){
        if(bestLabels[i]==0){
            Point2f c = points[i];
            circle( img, c, 10, colorTab[0], 1, LINE_AA );
        }
        if(bestLabels[i]==1){
            Point2f c = points[i];
            circle( img, c, 10, colorTab[1], 1, LINE_AA );
        }
        if(bestLabels[i]==2){
            Point2f c = points[i];
            circle( img, c, 10, colorTab[2], 1, LINE_AA );
        }
    }
    namedWindow("Clusters 2");
        imshow("Clusters 2", img);
        waitKey(0);
        cout<<points[0]<<"\n";
    */

    for(int i=0;i<centers.size();i++){
        int nord = centers[i].y;
        int sud = nord;
        int est = centers[i].x;
        int ovest = est;
        for(int j=0;j<points.size();j++){
            if(bestLabels[j]==i){
                if(points[j].x<est) est=points[j].x;
                if(points[j].x>ovest) ovest=points[j].x;
                if(points[j].y<nord) nord=points[j].y;
                if(points[j].y>sud) sud=points[j].y;
            }
        }
        Point2f ne=Point2f(est,nord);
        Point2f so=Point2f(ovest,sud);
        extremes.push_back(vector<Point2f>{ne,so});
        rectangle(img,ne,so,colorTab[i],3);
    }
    namedWindow("Bounding Box");
    imshow("Bounding Box", img);
    waitKey(0);
}

void refineBoundingBox(Mat& src, vector<Point2f>& inputCorners, vector<Point2f>& outputCorners, double MSradius, double threshold, double centroidRadius, bool showResult) {
    // We compute again the keypoints
    vector<KeyPoint> keypoints;
    Mat descriptors;
    featureDetector(src, keypoints, descriptors);

    // And run the mean shift inside the bounding box
    vector<vector<Point2f>> paths;
    meanShift_keypoints(src, keypoints, MSradius, threshold, paths);

    // And we compute the new centroids
    vector<Point2f> centroids;
    findCentroids(paths, centroidRadius, centroids);

    // In the vector paths we have as starting point the keypoint, and as final point the centroid

    if (showResult) {
        Mat displayPaths = src.clone();
        for (int i=0; i<paths.size(); i++) {
            drawPath(displayPaths, paths[i], Scalar(255,0,0));
        }
        for (int i=0; i<centroids.size(); i++) {
            circle(displayPaths, centroids[i], 10, Scalar(255,255,0), 3, 8, 0);
        }
        /*namedWindow("Keypoints");
        imshow("Keypoints", displayPaths);
        waitKey(0);*/
    }
    
    // We count, for each centroid, how many points converge to it. Then keep only the one with the highest count
    vector<int> centroidCounters;
    vector<int> labels(paths.size());
    for (int i=0; i<centroids.size(); i++) {
        // Counter that indicates how many keypoints converge to the centroid
        int counter = 0;

        // Scan the final point of the path and check if it coincides with the current centroid
        for (int j=0; j<paths.size(); j++) {
            if ( pointsDistance( paths[j].back(), centroids[i] ) < 0.1) {
                counter++;
                labels[j] = i;
            }
        }

        centroidCounters.push_back(counter);
    }
    for (int i=0; i<centroidCounters.size(); i++) {cout << centroidCounters[i] << endl;}
    //for (int i=0; i<labels.size(); i++) {cout << labels[i] << endl;}

    // Select the centroid with the highest count
    int bestCentroid;
    int max = 0;
    for (int i=0; i<centroidCounters.size(); i++) {
        if (centroidCounters[i] > max) {
            max = centroidCounters[i];
            bestCentroid = i;
        }
    }

    // Define the vector of selected keypoints (ie the ones converging to the best centroid)
    vector<Point2f> selectedKeypoints;
    for (int i=0; i<labels.size(); i++) {
        if (labels[i] == bestCentroid) {
            selectedKeypoints.push_back( paths[i].front() );
        }
    }

    // Draw the keypoints and highlight the ones converging to the best centroid
    if (showResult) {
        Mat displayKeypoints = src.clone();
        for (int i=0; i<paths.size(); i++) {
            circle(displayKeypoints, paths[i].front(), 10, Scalar(255,0,0));
        }
        for (int i=0; i<selectedKeypoints.size(); i++) {
            circle(displayKeypoints, selectedKeypoints[i], 10, Scalar(0,0,255));
        }
        namedWindow("Keypoints");
        imshow("Keypoints", displayKeypoints);
        waitKey(0);
    }

    // Redefine the bounding box
    double min_x = 1000000, min_y = 1000000;
    double max_x = 0, max_y = 0; 

    for (int i=0; i<selectedKeypoints.size(); i++) {
        Point2f p = selectedKeypoints[i];

        if (p.x < min_x) min_x = p.x;
        if (p.x > max_x) max_x = p.x;
        if (p.y < min_y) min_y = p.y;
        if (p.y > max_y) max_y = p.y;
    }

    double width = max_x - min_x;
    double height = max_y - min_y;

    double old_x = inputCorners[0].x;
    double old_y = inputCorners[0].y;

    Point2f upperLeftCorner( old_x+min_x, old_y+min_y );
    Point2f lowerRightCorner( old_x+min_x+width, old_y+min_y+height );

    outputCorners.push_back(upperLeftCorner);
    outputCorners.push_back(lowerRightCorner);    
}

void drawBoundingBoxes(Mat& img, vector<vector<Point2f>>& corners) {
    RNG rng(2092234);
    for (int i=0; i<corners.size(); i++) {
        Rect box(corners[i][0], corners[i][1]);
        rectangle(img, box, Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255)), 3, 8, 0);
    }
}