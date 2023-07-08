#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

void seg(Mat scr, Mat& dst, Vec3b color);
void regGrow(Mat scr, Mat& dst, Vec3b color);
void slideWind(Mat src, Mat &dst, int kerSize);
void grabAlg(Mat src, Mat& dst, vector<Point2f> bb);
void maskSovraposition(Mat src,Mat mask,Mat& dst);