#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "Detector.hpp"
#include "sliceMatch.h"
#include "Seg.hpp"
#include "BoundBox.hpp"

using namespace cv;
using namespace std;

void removeLowSaturationHSV(Mat& src, Mat& mask, double threshold) {

    // Assuming the src is passed as BGR image, we convert it to HSV
    Mat srcHSV;
    cvtColor(src, srcHSV, COLOR_BGR2HSV);

    // Select the saturation channel (the second)
    vector<Mat> channels;
    split(srcHSV, channels);
    Mat srcS = channels[1];

    // Remove all the pixel with saturation lower than the threshold and generate the mask
    inRange(srcS, threshold, 255, mask);
}

void maskSovraposition(Mat src, Mat mask, Mat& dst) {
    dst = src.clone();
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (mask.at<uchar>(i, j) == 0) {
                dst.at<Vec3b>(i, j)[0] = 0;
                dst.at<Vec3b>(i, j)[1] = 0;
                dst.at<Vec3b>(i, j)[2] = 0;
            }
        }
    }

}

// Se usate un IDE diverso da VSCode andate su gitignore.io e generate il vostro file gitignore specifico per il vostro IDE.
// Poi copiate e incollate sul file .gitignore che dovreste avere quando clonate la repo (se non c'e`, createlo)
// Tutto questo e` per evitare che vengano copiati file che non servono 

int main(int argc, char** argv) {
    for (int xx = 1; xx < 9;xx++) {
        cout << "tray number: " << xx << endl;
    Mat image = imread("../Food_leftover_dataset/tray"+to_string(xx)+"/food_image.jpg");
    Mat cleanImage = image.clone();
    vector<KeyPoint> keypoints;
    Mat descriptors;
    featureDetector(image, keypoints, descriptors);
    double radius = 200;
    double threshold = 0.5;
    vector<vector<Point2f>> paths;

    meanShift_keypoints(image, keypoints, radius, threshold, paths);

    double centroidRadius = 300;
    vector<Point2f> centroids;
    findCentroids(paths, centroidRadius, centroids);
    for (int i = 0; i < paths.size(); i++) {
        drawPath(image, paths[i], Scalar(255, 0, 0));
    }
    for (int i = 0; i < centroids.size(); i++) {
        circle(image, centroids[i], 50, Scalar(255, 0, 255), 3, 8, 0);
    }
    /// ELI
    /// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Scalar colorTab[] =
    {
        Scalar(0, 0, 255),
        Scalar(0,255,0),
        Scalar(255,100,100),
        Scalar(255,0,255),
        Scalar(0,255,255)
    };

    //To visualize the cluster centers

    Mat img = image.clone();
    for (int i = 0; i < (int)centroids.size(); ++i)
    {
        Point2f c = centroids[i];
        circle(img, c, 40, colorTab[i], 1, LINE_AA);
    }
    vector<int> labels(keypoints.size());
    vector<Point2f>points;
    for (int i = 0; i < keypoints.size(); i++) {
        points.push_back(keypoints[i].pt);
    }
    kMeans(centroids, points, labels, centroids.size());
    vector<Point2f> prunedPoints;
    vector<int> newLabels;
    double threshold2 = 325;
    int threshold3 = 100;
    clusterPruning(centroids, labels, threshold3);
    kMeans(centroids, points, labels, centroids.size());
    featurePruning(centroids, points, labels, prunedPoints, newLabels, threshold2);
    Mat finalImage = cleanImage.clone();
    for (int i = 0; i < prunedPoints.size(); ++i)
    {
        Point2f c = prunedPoints[i];
        if (newLabels[i] == 0)
            circle(finalImage, c, 10, colorTab[0], 1, LINE_AA);
        else if (newLabels[i] == 1)
            circle(finalImage, c, 10, colorTab[1], 1, LINE_AA);
        else
            circle(finalImage, c, 10, colorTab[2], 1, LINE_AA);
    }
    //Crea le Bounding Box
    vector<vector<Point2f>> extremes;
    BoundingBoxID(finalImage, prunedPoints, centroids, newLabels, extremes);

    vector<Point2f> finalPoints;
    vector<int> finalLabels;
    double threshold4 = 250;
    featurePruning(centroids, prunedPoints, newLabels, finalPoints, finalLabels, threshold4);
    extremes.clear();
    BoundingBoxID(finalImage, finalPoints, centroids, finalLabels, extremes);

    //Now let's devide the images
    vector<Mat> slicedImages;
    imageSlicer(extremes, cleanImage, slicedImages);

    //Testing segmantation

    vector<Mat> musk(slicedImages.size());


    Mat ggImg = cleanImage.clone();
    vector<vector<Point2f>> nExtremes;

    vector<Mat> mask(slicedImages.size());

    vector<Mat> finalMask(slicedImages.size());


    for (int i = 0; i < extremes.size(); i++) {
        grabAlg(ggImg, musk[i], extremes[i]);

        removeLowSaturationHSV(musk[i], mask[i], 30);


        maskSovraposition(ggImg, mask[i], finalMask[i]);

        RefinedBoxId(finalMask[i], nExtremes, centroids[i]);
    }
    vector<Mat> finalSlicedImg;
    imageSlicer(nExtremes, cleanImage, finalSlicedImg);
    String composition = "";
    for (int i = 0; i < centroids.size(); i++) {
        slicedImages[i] = colorReduce(finalSlicedImg[i],7);
        namedWindow("food", WINDOW_NORMAL);
        imshow("food", slicedImages[i]);
        slideClassifier(slicedImages[i], 10, false);
        //slideCounter(finalSlicedImg[i], 10);
        waitKey(0);
    }
    cout << "The plate contains: " <<composition << endl;
    destroyAllWindows();
}
}
