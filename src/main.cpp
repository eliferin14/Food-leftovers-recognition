#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "Detector.hpp"

using namespace cv;
using namespace std;

// Se usate un IDE diverso da VSCode andate su gitignore.io e generate il vostro file gitignore specifico per il vostro IDE.
// Poi copiate e incollate sul file .gitignore che dovreste avere quando clonate la repo (se non c'e`, createlo)
// Tutto questo e` per evitare che vengano copiati file che non servono 

int main(int argc, char** argv) {
    
    Mat image = imread("../Food_leftover_dataset/tray1/food_image.jpg");
    
    vector<KeyPoint> keypoints;
    Mat descriptors;
    featureDetector(image, keypoints, descriptors);

}