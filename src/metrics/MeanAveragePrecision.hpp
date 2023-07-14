#include <iostream>
#include "opencv2/opencv.hpp"
#include "DatasetLoader.hpp"

using namespace std;
using namespace cv;

// After loading the dataset, we have for each tray 2 objects of type TrayData
// One is the ground truth, the other is our detection

void getPrecisionRecall(BoundingBox& prediction, BoundingBox& truth, double& precision, double& recall);

// Metrics to evaluate the masks
void get_IoU(Mat& prediction, Mat& truth, double& iou);