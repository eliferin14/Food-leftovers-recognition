#include "MeanShift.hpp"
#include "BoundingBoxes.hpp"
#include "Masks.hpp"
#include "../utils.hpp"
#include "MaskSplitter.hpp"
#include "Estimator.hpp"
#include <opencv2/core/utils/filesystem.hpp>

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

void segmentImage(string filepath, string filename, string outputPath) {

    // Open the image
    Mat image = imread(filepath);

    cout << "Mean shift" << endl;

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
        //showImage("Centroids", paths_image);
        imwrite(outputPath+filename+"_centroid.jpg", paths_image);

    // Assign each keypoint to a centroid
    vector<vector<Point2f>> clusters;
    kmeansClustering(keypoints, centroids, clusters);
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
        //showImage("Clustered points", clusters_image2);
        imwrite(outputPath+filename+"_clustered_points.jpg", clusters_image2);

    // Cluster again, with fewer centroids
    kmeansClustering(keypoints, centroids, clusters);
    
    // Pruning
    cout << "Pruning" << endl;

    // Distance pruning
    double distanceThreshold = 250;
    distancePruning(clusters, centroids, distanceThreshold);

        Mat pruned_image = image.clone();
        for (int i=0; i<clusters.size(); i++) {
            circle(pruned_image, centroids[i], distanceThreshold, colorTab[i] );
        }


    // Gaussian pruning
    vector<Point2f> means;
    vector<Mat> covMatrices;
    double varianceThreshold = 5;
    vector<RotatedRect> ellipses;
    computeMean(clusters, means);
    computeVarianceMatrices(clusters, means, covMatrices);

    //gaussianClustering(keypoints, means, covMatrices, clusters);

    gaussianPruning(clusters, means, covMatrices, varianceThreshold, ellipses);

        for (int i=0; i<ellipses.size(); i++) {
            ellipse(pruned_image, ellipses[i], colorTab[i], 2, 8);

            //printf("X: %f, Y: %f, Angle: %f\n", ellipses[i].size.width/2, ellipses[i].size.height/2, 180 - ellipses[i].angle);

            for(int j=0; j<clusters[i].size(); j++) {
                circle(pruned_image, clusters[i][j], 10, colorTab[i]);
            }
        }
        //showImage("Gaussian pruning", pruned_image);
        imwrite(outputPath+filename+"_pruned_keypoints.jpg", pruned_image);
    

    // Now we have the clusterized points inside the clusters matrix
    // Each row of clusters contains all the points assigned to that cluster, namely a plate

    // Bounding boxes
    cout << "Bounding boxes" << endl;
    vector<Rect> boundingBoxes;
    getBoundingBoxes(clusters, boundingBoxes);

        Mat boundingBoxes_image = image.clone();
        for (int i=0; i<boundingBoxes.size(); i++) {
            rectangle(boundingBoxes_image, boundingBoxes[i], colorTab[i], 2, 8, 0);
        }
        //showImage("Bounding boxes", boundingBoxes_image);

    Mat imageHighSat = image.clone();
    double saturationThreshold = 35;
    // Given the boundingBoxes, run the grabCut algorithm to define the masks
    cout << "Grabcut masks" << endl;
    vector<Mat> masks;
    grabCutSegmentation(imageHighSat, boundingBoxes, masks);

    // Remove low saturation pixels
    for (int i=0; i<masks.size(); i++) {
        Mat tempMask;
        // First threshold using otsu
        removeLowSaturation_otsu(masks[i], tempMask);
        // Return to 3-channel BGR with the mask applied
        bitwise_and(image, image, masks[i], tempMask);
        // Second threshold with fixed value
        removeLowSaturation(masks[i], tempMask, saturationThreshold);
        //showImage("Maks", masks[i]);
        //showImage("Temp", tempMask);
        masks[i] = tempMask.clone();
    }

        for (int i=0; i<masks.size(); i++) {
            //showImage("Maks", masks[i]);
        }

    // Convert to binary mask and closing operation
    cout << "Mask postprocess" << endl;
    masksPostprocess(masks, filename);
    
        for (int i=0; i<masks.size(); i++) {
            //showImage("Maks", masks[i]);
        }

        for (int i=0; i<masks.size(); i++) {
            Mat maskBGR;
            bitwise_and(image, image, maskBGR, masks[i]);
            //showImage("Final mask BGR", maskBGR);
        }

    // Redefine the bounding boxes to be exactly as large as the masks
    refineBoundingBoxes(masks, boundingBoxes);

        boundingBoxes_image = image.clone();
        for (int i=0; i<boundingBoxes.size(); i++) {
            rectangle(boundingBoxes_image, boundingBoxes[i], colorTab[i], 2, 8, 0);
        }
        //showImage("Bounding boxes refined", boundingBoxes_image);

    // Draw contours on image
    Mat contours_image = boundingBoxes_image.clone();
    for (int i=0; i<masks.size(); i++) {
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(masks[i], contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
        for (size_t j=0; j<contours.size(); j++) {
            drawContours( contours_image, contours, (int)j, colorTab[i], 2, LINE_8, hierarchy, 0);
        }
    }
    //showImage("Contours", contours_image);
    imwrite(outputPath+filename+"_contours.jpg", contours_image);

    // Classification
    vector<int> labels(boundingBoxes.size(), -1);

    // Save the data

    // Save the cumulative mask
    string masksPath = outputPath + "masks/";
    Mat masksUnion;
    uniteMasks(masks, masksUnion);
    //showImage("Masks union", masksUnion);
    imwrite(masksPath+filename+"_masksUnion.jpg", masksUnion);

    // Save the single masks
    for (int i=0; i<masks.size(); i++) {
        // We want to save only what is inside the bounding box
        Mat colorMask;
        cv::bitwise_and(image, image, colorMask, masks[i]);
        Mat maskBB = colorMask(boundingBoxes[i]);
        //showImage("maskBB", maskBB);
        
        imwrite(masksPath + filename + "_masks/mask_" + to_string(i) + ".jpg", masks[i]);
        imwrite(masksPath + filename + "_masks/maskBB_" + to_string(i) + ".jpg", maskBB);
    }

    // Save the cumulative bounding box
    string bbPath = outputPath + "bounding_boxes/";
    //cout << bbPath << endl;
    saveBoundingBoxes(boundingBoxes, labels, bbPath+filename+"_bounding_box.txt");
    
}

void iterateDataset(string dataset_path, string output_dataset_path) {// Filenames of the images of each tray
    vector<string> filenames = {"food_image", "leftover1", "leftover2", "leftover3"};

    // Create the output dataset directory
    cv::utils::fs::createDirectory(output_dataset_path);

    // Iterate on the entire dataset
    for (int tray=1; tray<=8; tray++) {

        string tray_path = dataset_path + "/tray" + to_string(tray);
        string output_tray_path = output_dataset_path + "/tray" + to_string(tray) + "/";

        // Create folders
        cv::utils::fs::createDirectory(output_tray_path);
        cv::utils::fs::createDirectory(output_tray_path + "bounding_boxes");
        cv::utils::fs::createDirectory(output_tray_path + "masks");


        for (int i=0; i<filenames.size(); i++) {
            string filepath = tray_path + "/" + filenames[i] + ".jpg";
            cv::utils::fs::createDirectory(output_tray_path + "masks/" + filenames[i] + "_masks");


            cout << "======================================" << endl;
            cout << filepath << endl;
            cout << output_tray_path << endl;

            segmentImage(filepath, filenames[i], output_tray_path);
        }
    }
}

void compareTwoImages(string foodPath, string leftoverPath, string outputRootPath) {
    // Create the folder structure
    // We assume the folder indicated by outputRootPath already exists
    string outputPath = outputRootPath + "/comparison/";
    cv::utils::fs::createDirectory(outputPath);
    cv::utils::fs::createDirectory(outputPath + "bounding_boxes");
    cv::utils::fs::createDirectory(outputPath + "masks");
    cv::utils::fs::createDirectory(outputPath + "masks/" + "food_image" + "_masks");
    cv::utils::fs::createDirectory(outputPath + "masks/" + "leftover" + "_masks");


    // Analyze the food image
    cout << "======================================" << endl;
    cout << foodPath << endl;
    segmentImage(foodPath, "food_image", outputPath);

    // Analyze the leftover image
    cout << "======================================" << endl;
    cout << leftoverPath << endl;
    segmentImage(leftoverPath, "leftover", outputPath);

    cout << "\nThe output can be found in \"" << outputPath << "\"\n" << endl;
}

int main(int argc, char** argv) {

    if (argc < 2) {
        cerr << "Please choose \"dataset\" or \"twoImages\"" << endl;
        exit(-1);
    }

    if (!strcmp(argv[1], "dataset")) {
        // Datasets relative paths
        string dataset_path = "../Food_leftover_dataset"; 
        string output_dataset_path = "../Our_dataset";   

        iterateDataset(dataset_path, output_dataset_path);

        datasetEstimator(output_dataset_path);
    }
    else if (!strcmp(argv[1], "twoImages")) {
        if (argc != 4) {
            cerr << "Please provide two relative paths for the two images to be compared. The first is the tray with full plates, while the second is the tray with the leftovers" << endl;
            exit(-1);
        }

        compareTwoImages(argv[2], argv[3], "..");
        estimator();    

    }
    else {
        cerr << "Please choose \"dataset\" or \"twoImages\"" << endl;
        exit(-1);
    }
}