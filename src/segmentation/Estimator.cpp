#include "Estimator.hpp"

void estimator(){
    std::cout<<"=====================\nFood LeftOver estimation\n";
    cv::Mat foodMask = cv::imread("../comparison/masks/food_image_masksUnion.jpg");
    cv::Mat leftMask = cv::imread("../comparison/masks/leftover_masksUnion.jpg");
    float pixFood = countPixelOfMask(foodMask);
    float pixLeft = countPixelOfMask(leftMask);
    float ratio = pixLeft/pixFood;

    std::cout<<"Pixel in leftover image = "<<pixLeft<<"\nPixel in food image = "<<pixFood<<"\n\nRatio = "<<ratio<<"\n";

    std::ofstream of("../comparison/Food_leftover_estimation.txt");

    if (of.is_open()) {
        std::stringstream rowSS;
        rowSS << "Pixel in leftover image = "<<pixLeft<<"\nPixel in food image = "<<pixFood<<"\n\nRatio = "<<ratio<<"\n";
        std::string row = rowSS.str();
        of << row << '\n';
        // Close the file
        of.close();
    }
}

void datasetEstimator(std::string outputPath){
    std::cout<<"=====================\nFood LeftOver estimation\n";
    for(int i=1;i<=8;i++){
        std::string tray_path= outputPath + "/tray"+std::to_string(i);
        cv::Mat foodMask=cv::imread(tray_path+"/masks/food_image_masksUnion.jpg");
        float pixFood=countPixelOfMask(foodMask);
        for(int j=1;j<=3;j++){
            cv::Mat leftMask=cv::imread(tray_path+"/masks/leftover"+std::to_string(j)+"_masksUnion.jpg");
            float pixLeft=countPixelOfMask(leftMask);
            float ratio=pixLeft/pixFood;

            std::ofstream of(tray_path+"/Food_leftover_estimation"+std::to_string(i)+"_"+std::to_string(j)+".txt");

            if (of.is_open()) {
                std::stringstream rowSS;
                rowSS << "Pixel in leftover image = "<<pixLeft<<"\nPixel in food image = "<<pixFood<<"\n\nRatio = "<<ratio<<"\n";
                std::string row = rowSS.str();
                of << row << '\n';
                // Close the file
                of.close();
            }
        }
    }
}

