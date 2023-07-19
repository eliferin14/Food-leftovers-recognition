#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

//Doesn't work
void seg(Mat scr, Mat& dst, Vec3b color);
void regGrow(Mat scr, Mat& dst, Vec3b color);
void slideWind(Mat src, Mat &dst, int kerSize);

//Works
void grabAlg(Mat src, Mat& dst, vector<Point2f> bb);
void maskSovraposition(Mat src,Mat mask,Mat& dst);
void coloredMask(Mat cleanImg, Mat& dst, vector<Mat> mask, vector<vector<Point2f>> extremes);
void kmeanColor(Mat src,Mat& dst, int k);