#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

void waterSegmentation(Mat& img){
   /*
    Mat bwImg=img.clone();
    Mat edges = Mat::zeros(Size(img.cols,img.rows),CV_8UC1);
    cvtColor(img,bwImg,COLOR_BGR2GRAY);
    namedWindow("Pruned", WINDOW_NORMAL);
    imshow("Pruned", bwImg);
    waitKey(0);
    Canny(bwImg, edges, 100, 200);
    Mat tmp=img.clone();
    cout<<img.empty();
    cout<<img.type()<<"\n";
    cout<<edges.empty();
    cout<<edges.type()<<"\n";
    watershed(img, edges);
    namedWindow("Segmentation", WINDOW_NORMAL);
    imshow("Segmentation", edges);
    waitKey(0);
    */
    Mat bwImg=img.clone();
    Mat bgImg = bwImg.clone();
    Mat fgImg = bwImg.clone();
    Mat unk = bwImg.clone();
    Mat marker= bwImg.clone();
    cvtColor(img,bwImg,COLOR_BGR2GRAY);

    //Thresholding
    double thresh = threshold(bwImg, bwImg, 0, 255, THRESH_BINARY_INV+THRESH_OTSU);
    namedWindow("Thresholding", WINDOW_NORMAL);
    imshow("Thresholding", bwImg);
    waitKey(0);

    //noise removal
    Mat kernel = getStructuringElement(MORPH_RECT, Size_(3,3)); 
    morphologyEx(bwImg, bwImg, MORPH_OPEN, kernel, Point2f(-1,-1), 3);
    
    namedWindow("Noise Removal", WINDOW_NORMAL);
    imshow("Noise Removal", bwImg);
    waitKey(0);

    //Sure background
    dilate(bwImg, bgImg, kernel, Point2f(-1,-1), 3);
    namedWindow("Sure Background", WINDOW_NORMAL);
    imshow("Sure Background", bgImg);
    waitKey(0);

    //Sure foreground
    distanceTransform(bwImg, fgImg, DIST_L2, DIST_MASK_3);

    double min, max;
    minMaxIdx(fgImg, &min, &max);

    threshold(fgImg, fgImg, 0.7*max, 255, 0);
    namedWindow("Sure ForeGround", WINDOW_NORMAL);
    imshow("Sure Foreground", fgImg);
    waitKey(0);

    //Finding unknown regions
    //cout<<bgImg.type()<<" "<<fgImg.type()<<"\n";
    fgImg.convertTo(fgImg, CV_8UC1);
    // cout<<bgImg.type()<<" "<<fgImg.type()<<"\n";
    subtract(bgImg,fgImg,unk);
    namedWindow("Unknown regions", WINDOW_NORMAL);
    imshow("Unknown regions", unk);
    waitKey(0);

    // Marker labelling
    connectedComponents(fgImg, marker);

    marker+=1;
    
    marker.setTo(0, unk == 255);
    //marker[unk==255] = 0;
   

    //Watershed

    watershed(img,marker);
    img.setTo(Vec3b(255,0,0), marker==-1);
     namedWindow("Marker labelling", WINDOW_NORMAL);
    imshow("Marker labelling", marker);
    waitKey(0);

}