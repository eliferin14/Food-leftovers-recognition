#include "opencv2/opencv.hpp"
#include "gaborClassifier.h"

using namespace std;
using namespace cv;

cv::Mat getGaborKernel(double sigma, double theta, double lambda, double gamma)
{
    int kernelSize = static_cast<int>(8.0 * sigma); // Adjust kernel size based on sigma
    if (kernelSize % 2 == 0) // Ensure odd kernel size
        kernelSize++;

    cv::Mat kernel(kernelSize, kernelSize, CV_32F);
    double sigmaX = sigma;
    double sigmaY = sigma / gamma;
    double thetaRad = theta * CV_PI / 180.0;
    double psi = 0.0;
    double lambdaX = lambda;
    double lambdaY = lambda;

    double halfKernelSize = (kernelSize - 1) * 0.5;
    double cosTheta = std::cos(thetaRad);
    double sinTheta = std::sin(thetaRad);

    for (int x = -halfKernelSize; x <= halfKernelSize; ++x)
    {
        for (int y = -halfKernelSize; y <= halfKernelSize; ++y)
        {
            double xPrime = cosTheta * x + sinTheta * y;
            double yPrime = -sinTheta * x + cosTheta * y;

            double value = std::exp(-0.5 * (std::pow(xPrime / sigmaX, 2.0) + std::pow(yPrime / sigmaY, 2.0))) *
                std::cos(2.0 * CV_PI / lambdaX * xPrime + psi);

            kernel.at<float>(y + halfKernelSize, x + halfKernelSize) = value;
        }
    }

    return kernel;
}


int classifier(const Mat& filteredImage) {
    // Define classification rules here
    // Example rules:
    // Rule 1: If the mean intensity of the filtered image is high, classify as fruit
    // Rule 2: If the standard deviation of the filtered image is low, classify as baked good
    // Add more rules as needed

    double meanIntensity = cv::mean(filteredImage)[0];
    double stdDeviation = 0.0;
    cv::Scalar mean, stddev;
    cv::meanStdDev(filteredImage, mean, stddev);
    stdDeviation = stddev[0];

    if (meanIntensity > 150.0) {
        cout << "Class 1" << endl;
        cout << to_string(stdDeviation) << endl;
        cout << to_string(meanIntensity) << endl;
        return 1; // Classify as fruit
    }
    else if (stdDeviation < 30.0) {
        cout << "Class 2" << endl;
        cout << to_string(stdDeviation) << endl;
        cout << to_string(meanIntensity) << endl;
        return 2; // Classify as baked good
    }
    else {
        cout << "Class 3" << endl;
        cout << to_string(stdDeviation) << endl;
        cout << to_string(meanIntensity) << endl;
        return 0; // Classify as unknown/other
    }
}