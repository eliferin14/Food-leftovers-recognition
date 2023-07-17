#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "Detector.hpp"
#include "utils.hpp"

using namespace cv;
using namespace std;

// Se usate un IDE diverso da VSCode andate su gitignore.io e generate il vostro file gitignore specifico per il vostro IDE.
// Poi copiate e incollate sul file .gitignore che dovreste avere quando clonate la repo (se non c'e`, createlo)
// Tutto questo e` per evitare che vengano copiati file che non servono 

int main(int argc, char** argv) {
    
    Mat image = imread(argv[1]);
    Mat mask, masked;

    double otsuThreshold = removeLowSaturationHSV_otsu(image, mask);
    removeLowSaturationHSV(image, mask, otsuThreshold/3);

    bitwise_and(image, image, masked, mask);

    showImage("Before", image);
    showImage("After", masked);

}