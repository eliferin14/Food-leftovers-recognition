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



//Function to apply cv::grabCut function. Implementation of GrabAlgorithm
void grabAlg(Mat src,Mat& dst, vector<Point2f> bb){
    dst=src.clone();
    Mat mask;
    Mat backGround;
    Mat foreGround;
    int temp_x1, temp_x2, temp_y1, temp_y2;
    temp_x1 = bb[0].x;
    temp_y1 = bb[0].y;
    temp_x2 = bb[1].x - temp_x1;
    temp_y2 = bb[1].y - temp_y1;
    //cout<<"DENTRO GRABCUT x, width: "<<temp_x1<<", "<<temp_x2<<"\ny,height: "<<temp_y1<<", "<<temp_y2<<"\n";
    Rect ret=Rect(temp_x1,temp_y1,temp_x2,temp_y2);
    grabCut(dst,mask,ret, backGround,foreGround,5,GC_INIT_WITH_RECT);
    //drawing Foreground on dst
    for(int i=0;i<dst.rows;i++){
        for(int j=0;j<dst.cols;j++){
            if(mask.at<uchar>(i,j)==0 || mask.at<uchar>(i,j)==2){
                dst.at<Vec3b>(i,j)[0]=0;
                dst.at<Vec3b>(i,j)[1]=0;
                dst.at<Vec3b>(i,j)[2]=0;
            }
        }
    }
    //namedWindow("Mask",WINDOW_AUTOSIZE);
    //imshow("Mask",dst);
    //waitKey(0);
}

//Colored defined mask given a black and white mask.
void maskSovraposition(Mat src,Mat mask,Mat& dst){
    dst=src.clone();
    for(int i=0;i<src.rows;i++){
        for(int j=0;j<src.cols;j++){
            if(mask.at<uchar>(i,j)==0){
                dst.at<Vec3b>(i,j)[0]=0;
                dst.at<Vec3b>(i,j)[1]=0;
                dst.at<Vec3b>(i,j)[2]=0;
            }
        }
    }

}

void coloredMask(Mat cleanImg, Mat& dst, vector<Mat> mask, vector<vector<Point2f>> extremes){
    dst=Mat(cleanImg.rows, cleanImg.cols, CV_8UC1, Scalar(0));
    //coloring only masks
    vector<int> colors={40, 80, 120 , 160, 200, 240, 255};
    for(int i=0;i<extremes.size();i++){
        for(int x=extremes[i][0].x;x<=extremes[i][1].x;x++){
            for(int y=extremes[i][0].y;y<=extremes[i][1].y;y++){
                if(mask[i].at<uchar>(y,x)!=0){//Before it was mask
                    dst.at<uchar>(y,x)=colors[i];
                    //darkImg.at<Vec3b>(y,x)[1]=colors[i];
                    //darkImg.at<Vec3b>(y,x)[2]=colors[i];
                }
            }
        }
    }
    namedWindow("ColoredMask",WINDOW_AUTOSIZE);
    imshow("ColoredMask",dst);
    waitKey(0);

}

void kmeanColor(Mat src,Mat& dst, int k){
    dst=src.clone();
    Mat samples(src.rows*src.cols, src.channels(), CV_32F); //change to float
    for(int x=0;x<src.cols;x++){
        for(int y=0;y<src.rows;y++){
            for(int c=0;c<src.channels();c++){
                samples.at<float>(y+x*src.rows,c)=src.at<Vec3b>(y,x)[c];
            }
        }
    }

    Mat labels; 
    Mat centers;
    int iter=5;
    kmeans(samples,k,labels,TermCriteria(TermCriteria::MAX_ITER|TermCriteria::EPS,10,1.0), iter,KMEANS_PP_CENTERS,centers);

    for(int x=0;x<src.cols;x++){
        for(int y=0;y<src.rows;y++){
            int ind=labels.at<int>(y+x*src.rows, 0);
            for(int c=0;c<src.channels();c++){
                dst.at<Vec3b>(y,x)[c]=centers.at<float>(ind,c);
            }
        }
    }

}
