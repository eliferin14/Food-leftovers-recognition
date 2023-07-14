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
    void loadBoundingBoxes();
    void loadMasks();
    friend ostream& operator<<(ostream& os, const TrayData& tray);
};

// Helper functions
void loadLabels_singleTray(string filename, vector<int>& labels);
void loadLabels(string foldername, vector<vector<int>>& labels);

#endif