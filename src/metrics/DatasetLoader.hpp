// Set of function we use to load the dataset and extract all the information we need to compute the metrics, namely the labels, the bounding boxes and the masks
// Can be used for both the ground truth and our predicitions, assuming they have the same structure

// The labels for each tray are stored in the file "labels.txt" inside each tray's folder

#ifndef DS_LOADER
#define DS_LOADER

#include <iostream>
#include "opencv2/opencv.hpp"
#include <fstream>

using namespace std;
using namespace cv;

// Contains all the information we need about a bounding box, namely its position, dimension, and label of what it contains
class BoundingBox {
public:
    int label;
    Rect boundingBox;
    BoundingBox(int l, Rect bb);
    friend ostream& operator<<(ostream& os, const BoundingBox& bb);
};

// Contains all the information related to a SINGLE tray
class TrayData {
public:
    string path;
    vector<int> foodLabels;
    vector<vector<BoundingBox>> boundingBoxes;
    vector<vector<Mat>> masks;

    TrayData(string path);
    void loadLabels();
    void loadBoundingBoxes(); // CHECK SEGMENTATION FAULT SUL VECTOR CREATO DENTRO
    void loadMasks();
    friend ostream& operator<<(ostream& os, const TrayData& tray);
};

// Helper functions
void loadLabels_singleTray(string filename, vector<int>& labels);
void loadLabels(string foldername, vector<vector<int>>& labels);

// Load our bounding boxes
// Needs to be different because we had problems with classification
void loadOurBoundingBoxes_singleFile(std::string filepath, std::vector<cv::Rect>& boundingBoxes);
void loadOurBoundingBoxes_singleTray(std::string trayPath, std::vector<std::vector<cv::Rect>>& trayBoundingBoxes);
void loadOurBoundingBoxes(std::string ourDatasetPath, std::vector<std::vector<std::vector<cv::Rect>>>& boundingBoxes);

// Load dataset's bounding boxes
void loadTrueBoundingBoxes_singleFile(std::string filepath, std::vector<cv::Rect>& boundingBoxes);
void loadTrueBoundingBoxes_singleTray(std::string trayPath, std::vector<std::vector<cv::Rect>>& trayBoundingBoxes);
void loadBoundingBoxes(std::string ourDatasetPath, std::vector<std::vector<std::vector<cv::Rect>>>& boundingBoxes);

// Load our masks
void loadOurMasks_singleTray(std::string trayPath, std::vector<cv::Mat>& trayMasks);
void loadOurMasks(std::string ourDatasetPath, std::vector<std::vector<cv::Mat>>& masks);

// Load the true masks
void loadMasks_singleTray(std::string trayPath, std::vector<cv::Mat>& trayMasks, bool flag);
void loadMasks(std::string trueDatasetPath, std::vector<std::vector<cv::Mat>>& masks, bool flag);

#endif