#include "MeanAveragePrecision.hpp"

void getPrecisionRecall(BoundingBox& prediction, BoundingBox& truth, double& precision, double& recall) {
    // Check if the bounding boxes are referred to the same food
    if (prediction.label != truth.label) {
        cerr << "The two bounding boxes are related to two different foods!" << endl;
        exit(0);
    }

    Rect pRect = prediction.boundingBox;
    Rect tRect = truth.boundingBox;

    // First we compute the true positives, the true negatives and the false negatives

    // Get the intersection of the two rectangles

    // Upper left corner
    int ix1 = max(pRect.x, tRect.x);
    int iy1 = max(pRect.y, tRect.y);

    // Lower right corner
    int ix2 = min(pRect.br().x, tRect.br().x);
    int iy2 = min(pRect.br().y, tRect.br().y);

    //printf("Prediction BB: (%d, %d), (%d, %d)\n", pRect.x, pRect.y, pRect.br().x, pRect.br().y);
    //printf("Truth BB: (%d, %d), (%d, %d)\n", tRect.x, tRect.y, tRect.br().x, tRect.br().y);
    //printf("Intersection: (%d, %d), (%d, %d)\n", ix1, iy1, ix2, iy2);

    // Compute intersection width, height and area (True Positives)
    int iWidth = max(ix2-ix1, 0);
    int iHeight = max(iy2-iy1, 0);
    double TP = iWidth * iHeight;
    //...
}

void get_IoU(Mat& prediction, Mat& truth, double& iou) {
    int width = prediction.size().width;
    int height = prediction.size().height;

    // Check each pixel and update the counters
    int TP = 0;
    int FP = 0;
    int FN = 0;
    for(int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            // Look at the pixel in the two masks
            bool p = ( prediction.at<uchar>(i,j) > 0 );
            bool t = ( truth.at<uchar>(i,j) > 0 );

            // Update the counter accordingly
            if ( p && t ) TP++;
            else if ( p && !t ) FP++;
            else if ( !p && t ) FN++;
        }
    }

    // Compute the IoU score
    iou = (double)TP / (TP + FP + FN);
}