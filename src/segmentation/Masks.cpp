#include "Masks.hpp"

void grabCutSegmentation(cv::Mat& src, std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& masks) {

    for (int i=0; i<boundingBoxes.size(); i++) {
        cv::Mat mask, background, foreground;

        cv::grabCut( src, mask, boundingBoxes[i], background, foreground, 5, cv::GC_INIT_WITH_RECT);

        for (int x=0; x<mask.cols; x++) {
            for (int y=0; y<mask.rows; y++) {
                if (mask.at<uint8_t>(y,x) == 2) mask.at<uint8_t>(y,x)=0;
            }
        }

        cv::Mat mask2;
        cv::bitwise_and(src, src, mask2, mask);

        masks.push_back(mask2);
    }

}

void removeLowSaturation(cv::Mat& src, cv::Mat& dst, double threshold) {

	// Assuming the src is passed as BGR image, we convert it to HSV
	cv::Mat srcHSV, mask;
	cv::cvtColor(src, srcHSV, cv::COLOR_BGR2HSV);

	// Select the saturation channel (the second)
	std::vector<cv::Mat> channels;
	cv::split(srcHSV, channels);
	cv::Mat srcS = channels[1];

	// Remove all the pixel with saturation lower than the threshold and generate the mask
	cv::inRange(srcS, threshold, 255, mask);
    //cv::GaussianBlur(srcS, srcS, cv::Size(5,5), 0); cv::threshold(srcS, mask, 0, 255, cv::THRESH_OTSU);

    // Apply the mask to the source
    cv::bitwise_and(src, src, dst, mask);
}

void removeLowSaturation_otsu(cv::Mat& src, cv::Mat& dst) {
    // Assuming the src is passed as BGR image, we convert it to HSV
	cv::Mat srcHSV;
	cv::cvtColor(src, srcHSV, cv::COLOR_BGR2HSV);

	// Select the saturation channel (the second)
	std::vector<cv::Mat> channels;
	cv::split(srcHSV, channels);
	cv::Mat srcS = channels[1];
    
    // Blur?
    cv::GaussianBlur(srcS, srcS, cv::Size(5,5), 0);

    // Otsu threshold
    long double otsuThreshold = threshold(srcS, dst, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
}

void refineBoundingBoxes(std::vector<cv::Mat>& masks, std::vector<cv::Rect>& boundingBoxes) {
    boundingBoxes.clear();

    for (int i=0; i<masks.size(); i++) {
        // Scan the mask and find the extremes
        double minX = 100000;
        double minY = 100000;
        double maxX = -1;
        double maxY = -1;

        cv::Mat maskGray = masks[i];

        for (int r=0; r<maskGray.rows; r++) {
            for (int c=0; c<maskGray.cols; c++) {
                // Consider only the pixels that are not black
                int val = maskGray.at<uint8_t>(r,c);
                if (val > 0) {
                    if (r < minY) minY = r;
                    if (r > maxY) maxY = r;
                    if (c < minX) minX = c;
                    if (c > maxX) maxX = c;
                }
            }
        }

        cv::Point2f upperLeftCorner(minX, minY);
        cv::Point2f lowerRightCorner(maxX, maxY);

        cv::Rect boundingBox(upperLeftCorner, lowerRightCorner);
        boundingBoxes.push_back(boundingBox);
    }
}

void masksPostprocess(std::vector<cv::Mat>& masks, std::string filename) {
    for (int i=0; i<masks.size(); i++) {
        // Convert to grayscale
        cv::Mat maskGray;
        cv::cvtColor(masks[i], maskGray, cv::COLOR_BGR2GRAY);

        // Binarization
        cv::inRange(maskGray, 1, 255, maskGray);

        // Closing operation only if not leftover
        if (!filename.compare("food_image")) {
            //std::cout << "Closing" << std::endl;
            morphologyEx(maskGray, maskGray, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(30,30)));
        }

        // Save the mask
        masks[i] = maskGray;
    }
}

void uniteMasks(std::vector<cv::Mat>& masks, cv::Mat& dst) {
    dst = masks[0].clone();
    for (int i=0; i<masks.size(); i++) {
        cv::bitwise_or(dst, masks[i], dst);
    }
}