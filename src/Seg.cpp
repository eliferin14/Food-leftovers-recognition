#include "opencv2/opencv.hpp"
#include "Seg.hpp"

using namespace std;
using namespace cv;

//Constant definition
//Bilateral Filtering parameters
const int kNeighborhoodDiamter = 5;
const int kSigmaColor = 1000;
const int kSigmaSpace = 200;

//Mask parameters


//Normalization parameters
const double kLowNorm = 0;
const double kHighNorm = 1.0;

//Thresholding parameters
const double kLowTresholdNorm = 0.237;
const double kHighThresholdNorm = 1.0;

//Circle parameters
const int kCircleRadius = 3;

void seg(Mat scr, Mat& dst, Vec3b color){
    
    //PREPROCCESSSING
    //dst=scr.clone();
    bilateralFilter(scr, dst, kNeighborhoodDiamter, kSigmaColor, kSigmaSpace);
    Mat mask;

    int offset=150;
    const Scalar kLowTreshColor = Scalar(0, 0, 0);
    const Scalar kHighTreshColor = Scalar(180, 180, 180);//color[0]+offset, color[1]+offset, color[2]+offset
    const Scalar kAssignedScalar = Scalar(255, 255, 255);

    inRange(dst, kLowTreshColor, kHighTreshColor, mask);
    dst.setTo(kAssignedScalar, mask);

    Mat distImg; //Forse da usare, per ora continuamo a sovrascrivere dst
    distanceTransform(mask, distImg, DIST_L2, 3);
    
    //Normalization
    Mat norImg;
    normalize(distImg, norImg, kLowNorm, kHighNorm, NORM_MINMAX);

    //Thresholding
    Mat threshImg;
    threshold(norImg, threshImg, kLowTresholdNorm,kHighThresholdNorm, THRESH_BINARY);

    //Conversion to CV_8U
    Mat dImg8;
    threshImg.convertTo(dImg8, CV_8U);

    //Find markers
    vector<vector<Point>> cVec;
    findContours(dImg8, cVec, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    //Markers image
    Mat markImg = Mat::zeros(threshImg.size(), CV_32S);
    for(int i=0; i<cVec.size();i++){
        drawContours(markImg, cVec, i, Scalar(i+1), -1);
    }

    circle(markImg, Point(5,5), kCircleRadius, Scalar(255), -1);

    //WATERSHADING :)
    cvtColor(dst, dst, COLOR_GRAY2BGR);
    watershed(dst, markImg);
    //ENDING OF WATERSHADIN :(
    
    //POST PROCCESSSING...or war syndrom
    Mat mark; //Zuckinenberg
    markImg.convertTo(mark, CV_8U);
    bitwise_not(mark,mark);

    vector<Vec3b> colors;
	for (size_t i = 0; i < cVec.size(); i++)
	{
		int b = theRNG().uniform(0, 255);
		int g = theRNG().uniform(0, 255);
		int r = theRNG().uniform(0, 255);
		colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
	}

    dst=Mat::zeros(markImg.size(), CV_8UC3);//DUCE??

    //WAAA Coloriamo tutto in maniera diversa
    for(int x=0; x<markImg.rows;x++){
        for(int y=0;y<markImg.cols;y++){
            int i=markImg.at<int>(x,y);
            if(i>0 && i< static_cast<int>(cVec.size()))
                dst.at<Vec3b>(x,y) = colors[i-1];
        }   
    }

    imshow("Segmented", dst);
    waitKey(0);

}

void regGrow(Mat scr, Mat& dst, Vec3b color){
 //
}

void slideWind(Mat src, Mat& dst, int kerSize){
    /*for(int i=0;i<scr.rows;i++)
    {
        for(int j=0;j<scr.cols;j++){

        }
    }*/
    dst=src.clone();
    //cvtColor(src, src,COLOR_BGR2GRAY);
    Mat kernel1 = Mat(kerSize, kerSize, CV_8U, Scalar(2));
    //blur(src,dst, Size(kerSize, kerSize));
    
	bilateralFilter(src, dst, -1, 200,20);

}