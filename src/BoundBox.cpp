#include "BoundBox.hpp"

using namespace std;
using namespace cv;

void BoundingBoxID(Mat& src, std::vector<Point2f> points, std::vector<Point2f> centers, vector<int> bestLabels, vector<vector<Point2f>>& extremes){

    Mat img=src.clone();

    Scalar colorTab[] =
    {
        Scalar(0, 0, 255),
        Scalar(0,255,0),
        Scalar(255,100,100),
        Scalar(255,0,255),
        Scalar(0,255,255)
    };
    /*
    for(int i=0;i<bestLabels.size();i++){
        if(bestLabels[i]==0){
            Point2f c = points[i];
            circle( img, c, 10, colorTab[0], 1, LINE_AA );
        }
        if(bestLabels[i]==1){
            Point2f c = points[i];
            circle( img, c, 10, colorTab[1], 1, LINE_AA );
        }
        if(bestLabels[i]==2){
            Point2f c = points[i];
            circle( img, c, 10, colorTab[2], 1, LINE_AA );
        }
    }
    namedWindow("Clusters 2");
        imshow("Clusters 2", img);
        waitKey(0);
        cout<<points[0]<<"\n";
    */

    for(int i=0;i<centers.size();i++){
        int nord = centers[i].y;
        int sud = nord;
        int est = centers[i].x;
        int ovest = est;
        for(int j=0;j<points.size();j++){
            if(bestLabels[j]==i){
                if(points[j].x<est) est=points[j].x;
                if(points[j].x>ovest) ovest=points[j].x;
                if(points[j].y<nord) nord=points[j].y;
                if(points[j].y>sud) sud=points[j].y;
            }
        }
        Point2f ne=Point2f(est,nord);
        Point2f so=Point2f(ovest,sud);
        extremes.push_back(vector<Point2f>{ne,so});
        rectangle(img,ne,so,colorTab[i],3);
    }
    namedWindow("Bounding Box",WINDOW_AUTOSIZE);
    imshow("Bounding Box", img);
    waitKey(0);
}

void RefinedBoxId(Mat& src, vector<vector<Point2f>>& extremes, Point2f centers){
    
    int nord = centers.y;
    int sud = nord;
    int est = centers.x;
    int ovest = est;
    
    Scalar colorTab[] =
    {
        Scalar(0, 0, 255),
        Scalar(0,255,0),
        Scalar(255,100,100),
        Scalar(255,0,255),
        Scalar(0,255,255)
    };
    
    for(int i=0;i<src.rows;i++){
        for(int j=0;j<src.cols;j++){
            if(src.at<Vec3b>(i,j)[0]!=0 && src.at<Vec3b>(i,j)[1]!=0 && src.at<Vec3b>(i,j)[2]!=0){
                if(i>sud) sud=i;
                if(i<nord) nord=i;
                if(j<est) est=j;
                if(j>ovest) ovest=j;
            }
        }
    }
    Point2f ne=Point2f(est,nord);
    Point2f so=Point2f(ovest,sud);
    extremes.push_back(vector<Point2f>{ne,so});

    rectangle(src,ne,so,Scalar(255,100,100),3);
    namedWindow("Clusters 2");
    imshow("Clusters 2", src);
    waitKey(0);

}