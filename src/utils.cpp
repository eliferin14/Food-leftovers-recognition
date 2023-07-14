#include "utils.hpp"

void showImage(std::string windowName, cv::Mat img) {
    cv::namedWindow(windowName);
    cv::imshow(windowName, img);
    cv::waitKey(0);
}

void drawHistogram(cv::Mat& src, cv::Mat& hist, int hist_w, int hist_h, int bins) {
    int histSize[] = {bins};

    int channels[] = {0};

    float graylevel_range[] = {0, 256};
    const float* ranges[] = {graylevel_range};

    cv::calcHist(&src, 1, channels, cv::Mat(), hist, 1, histSize, ranges);

    // Draw the histogram
    cv::Mat histImage = cv::Mat(hist_h, hist_w, CV_8U, cv::Scalar(255,255,255));

    // Find the max frequency in the bins
    double maxVal;
    cv::minMaxLoc(hist, 0, &maxVal, 0, 0);

    int bin_w = cvRound((float)hist_w / bins);  // pixel width of each bin

    for (int bin=0; bin<bins; bin++) {
        float binVal = hist.at<float>(bin); // Frequency of the graylevel

        for (int i=0; i<bin_w; i++) {
            int col_h = cvRound(binVal/maxVal*hist_h);  // Height of the column to draw, normalized to fit in the picture

            for (int j=0; j<col_h; j++) {
                histImage.at<unsigned char>(hist_h - col_h + j, bin*bin_w + i) = 0;
            }
            
        }
        
    }

    hist = histImage;
}


void showHistogram(std::string windowName, cv::Mat& src) {
    cv::Mat hist;
    drawHistogram(src, hist, 512, 256, 256);
    showImage(windowName, hist);
}