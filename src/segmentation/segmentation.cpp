#include "MeanShift.hpp"
#include "BoundingBoxes.hpp"
#include "Masks.hpp"
#include "../utils.hpp"

using namespace std;
using namespace cv;

vector<Scalar> colorTab =
    {
        Scalar(255,0,0),
        Scalar(0,255,0),
        Scalar(0,0,255),
        Scalar(255,0,255),
        Scalar(255,255,0),
        Scalar(0,255,255),
    };

void segmentImage(string filename, string path) {
    Mat image = imread(filename);

    // Detect keypoints with SIFT
    vector<KeyPoint> keypoints;
    Mat descriptors;
    featureDetector(image, keypoints, descriptors);

    // Mean Shift 
    double radius = 200;
    double threshold = 0.5;
    vector<vector<Point2f>> paths;
    meanShift_keypoints(image, keypoints, radius, threshold, paths);

    // Collapse the final points that are close to each other into one point
    double centroidRadius = 200;
    vector<Point2f> centroids;
    findCentroids(paths, centroidRadius, centroids);

        Mat paths_image = image.clone();
        for (int i=0; i<paths.size(); i++) {
            drawPath(paths_image, paths[i]);
        }
        showImage("Centroids", paths_image);

    // Assign each keypoint to a centroid
    vector<int> labels;
    vector<vector<Point2f>> clusters;
    assignLabels(centroids, paths, labels, clusters);
    /*
        Mat labels_image = image.clone();
        for ( int i=0; i<keypoints.size(); i++) {
            int label = labels[i];
            if (label < 0) label = colorTab.size();
            circle(labels_image, keypoints[i].pt, 10, colorTab[labels[i]]);
        }
        showImage("Labels", labels_image);

        Mat clusters_image = image.clone();
        for (int i=0; i<clusters.size(); i++) {
            for (int j=0; j<clusters[i].size(); j++) {
                circle(clusters_image, clusters[i][j], 10, colorTab[i]);
            }
        }
        showImage("Clustered points", clusters_image);
    */
    
    // If a centroid has less that {threshold} keypoints assigned to it, we remove it
    int clusterThreshold = 100;
    removeLowCountClusters(clusters, centroids, clusterThreshold);
    
        Mat clusters_image2 = image.clone();
        for (int i=0; i<clusters.size(); i++) {
            for (int j=0; j<clusters[i].size(); j++) {
                circle(clusters_image2, clusters[i][j], 10, colorTab[i]);
            }
        }
        showImage("Clustered points", clusters_image2);
    
    // Compute mean and variance of each cluster
    vector<Point2f> means;
    vector<Mat> covMatrices;
    double varianceThreshold = 5.5;
    vector<RotatedRect> ellipses;
    computeMean(clusters, means);
    computeVarianceMatrices(clusters, means, covMatrices);
    gaussianPruning(clusters, means, covMatrices, varianceThreshold, ellipses);
/*
        Mat gaussian_image = image.clone();
        for (int i=0; i<clusters.size(); i++) {
            ellipse(gaussian_image, ellipses[i], colorTab[i], 2, 8);
            
            Point2f vertices[4];
            ellipses[i].points(vertices);
            for (int j = 0; j < 4; j++)
            line(gaussian_image, vertices[j], vertices[(j+1)%4], colorTab[i], 2);

            printf("X: %f, Y: %f, Angle: %f\n", ellipses[i].size.width/2, ellipses[i].size.height/2, 180 - ellipses[i].angle);

            for(int j=0; j<clusters[i].size(); j++) {
                circle(gaussian_image, clusters[i][j], 10, colorTab[i]);
            }
        }
        showImage("Gaussian pruning", gaussian_image);
    */

    // Pruning di Crisci

    // Now we have the clusterized points inside the clusters matrix
    // Each row of clusters contains all the points assigned to that cluster, namely a plate

    // Bounding boxes
    vector<Rect> boundingBoxes;
    getBoundingBoxes(clusters, boundingBoxes);

        Mat boundingBoxes_image = image.clone();
        for (int i=0; i<boundingBoxes.size(); i++) {
            rectangle(boundingBoxes_image, boundingBoxes[i], colorTab[i], 2, 8, 0);
        }
        showImage("Bounding boxes", boundingBoxes_image);

    // Extract the submatrices (useless?)
    vector<Mat> platesBB;
    extractPlatesBB(image, boundingBoxes, platesBB);

        for (int i=0; i<platesBB.size(); i++) {
            //showImage("Plate", platesBB[i]);
        }

    // Remove low saturation pixels
    Mat imageHighSat = image.clone();
    double saturationThreshold = 30;
    removeLowSaturation(image, imageHighSat, saturationThreshold);
    // Given the boundingBoxes, run the grabCut algorithm to define the masks
    vector<Mat> masks;
    grabCutSegmentation(imageHighSat, boundingBoxes, masks);
    for (int i=0; i<masks.size(); i++) removeLowSaturation(masks[i], masks[i], saturationThreshold);

        for (int i=0; i<masks.size(); i++) {
            showImage("Maks", masks[i]);
        }

    // Convert to binary mask and closing operation
    masksPostprocess(masks);
    
        for (int i=0; i<masks.size(); i++) {
            showImage("Maks", masks[i]);
        }

    // Redefine the bounding boxes to be exactly as large as the masks
    refineBoundingBoxes(masks, boundingBoxes);

        boundingBoxes_image = image.clone();
        for (int i=0; i<boundingBoxes.size(); i++) {
            rectangle(boundingBoxes_image, boundingBoxes[i], colorTab[i], 2, 8, 0);
        }
        showImage("Bounding boxes refined", boundingBoxes_image);

    // Draw contours on image
    Mat contours_image = boundingBoxes_image.clone();
    printf("Channels: %d\n", masks[0].channels());
    for (int i=0; i<masks.size(); i++) {
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(masks[i], contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
        for (size_t j=0; j<contours.size(); j++) {
            drawContours( contours_image, contours, (int)j, colorTab[i], 2, LINE_8, hierarchy, 0);
        }
    }
    showImage("Contours", contours_image);
}

int main(int argc, char** argv) {

    string filename = argv[1];
    segmentImage(filename, "");

    return 0;
}