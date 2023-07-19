#include "MeanShift.hpp"

void featureDetector(cv::Mat& src, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {

    // SIFT parameters
    int nFeatures = 0;                  // 0
    int nOctaveLayers = 3;              // 3
    double contrastThreshold = 0.05;    // 0.04
    double edgeThreshold = 5;          // 10
    double sigma = 1.6;                 // 1.6

    // Feature detector 
    cv::Ptr<cv::SIFT> detectorPtr = cv::SIFT::create(nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);

    detectorPtr->detect(src, keypoints);
    detectorPtr->compute(src, keypoints, descriptors);
}

double pointsDistance(cv::Point p1, cv::Point p2) {
    float deltaX = p1.x - p2.x;
    float deltaY = p1.y - p2.y;

    float distance = sqrt(pow(deltaX, 2) + pow(deltaY, 2));

    return distance;
}

void getNeighbourhood(std::vector<cv::KeyPoint>& keypoints, cv::Point center, double radius, std::vector<cv::KeyPoint>& neighbourhood) {
    // Scan all the keypoints and select the ones with distance from the center < radius
    for ( int i=0; i<keypoints.size(); i++ ) {
        if ( pointsDistance( center, keypoints[i].pt ) <= radius ) {
            neighbourhood.push_back(keypoints[i]);
        }
    }
}

cv::Point2f getBaricenter(std::vector<cv::KeyPoint>& neighbourhood) {
    // If there are no neighbours, return (-1,-1);
    if (neighbourhood.size()==0) {
        return cv::Point2f(-1,-1);
    }

    // Compute the mean x and y of all the points
    float sumX=0, sumY=0;
    for (int i=0; i<neighbourhood.size(); i++) {
        sumX += neighbourhood[i].pt.x;
        sumY += neighbourhood[i].pt.y;
    }

    double barX = sumX / neighbourhood.size();
    double barY = sumY / neighbourhood.size();

    cv::Point2f baricenter(barX, barY);

    return baricenter;
}

void meanShift_onePoint(std::vector<cv::KeyPoint>& keypoints, cv::Point2f startingPoint, double radius, double threshold, std::vector<cv::Point2f>& baricenterIterations) {
    cv::Point2f center = startingPoint;
    baricenterIterations.push_back(center);
    double baricenterDisplacement = radius;
    
    // Iterate until the displacement between iterations is larger than the given threshold
    while (baricenterDisplacement >= threshold) {
        // Find the keypoints in the neighbourhood
        std::vector<cv::KeyPoint> neighbours;
        getNeighbourhood(keypoints, center, radius, neighbours);

        // Update the baricenter
        cv::Point2f oldCenter = center;
        center = getBaricenter(neighbours);

        // If there are no neighbours (baricenter is (-1,-1)), end the cycle
        if (center.x == -1 && center.y == -1) {
            return;
        }

        baricenterIterations.push_back(center);

        // Compute the displacement
        baricenterDisplacement = pointsDistance(center, oldCenter);
    }
}

void meanShift_keypoints(cv::Mat& src, std::vector<cv::KeyPoint>& keypoints, double radius, double threshold, std::vector<std::vector<cv::Point2f>>& paths) {
    for (int i=0; i<keypoints.size(); i++) {
        std::vector<cv::Point2f> path;
        meanShift_onePoint(keypoints, keypoints[i].pt, radius, threshold, path);
        paths.push_back(path);
    }
}

void findCentroids(std::vector<std::vector<cv::Point2f>>& paths, double radius, std::vector<cv::Point2f>& centroids) {
    // Scan all the paths
    for (int i=0; i<paths.size(); i++) {

        // If the path has only 1 point, ignore it
        if (paths[i].size()==1) continue;

        // Otherwise we select the final point
        cv::Point2f finalPoint = paths[i].back();

        // We check if the final point is close to a point that is already a centroid
        // If there are no close centroids, we include it in the std::vector. Otherwise we ignore it
        bool isClose = false;
        for (int j=0; j<centroids.size(); j++) {
            if (pointsDistance(centroids[j], finalPoint) < radius) {
                isClose = true;
                paths[i].push_back(centroids[j]);
                break;
            }
        }
        if ( !isClose ) centroids.push_back(finalPoint);
    }
}

void drawPath(cv::Mat& img, std::vector<cv::Point2f>& points) {
    if ( points.size() == 1 ) {
        return;
    }

    // Draw the starting point in green
    circle(img, points.front(), 3, cv::Scalar(0,255,0), 3, 8, 0);

    // Blue arrows between iterations
    for (int i=0; i<points.size()-1; i++) {
        arrowedLine(img, points[i], points[i+1], cv::Scalar(255,0,0), 2, 8, 0, 0.1);
    }

    // Draw the end point in red
    circle(img, points.back(), 3, cv::Scalar(0,0,255), 3, 8, 0);
}


void clusterize(std::vector<cv::Point2f>& centroids, std::vector<std::vector<cv::Point2f>>& paths, std::vector<std::vector<cv::Point2f>>& clusters) {
    // Initialize labels with -1 as default value
    std::vector<int> labels(paths.size(), -1);

    clusters = std::vector<std::vector<cv::Point2f>>(centroids.size());
    
    // For each starting keypoint, we look at the final point of the path and assign a label accordingly
    for (int i=0; i<paths.size(); i++) {

        // Select the final point
        cv::Point2f finalPoint = paths[i].back();
        
        // Assign it to a centroid
        for (int j=0; j<centroids.size(); j++) {

            // If close, the label is the index of the centroid
            if (pointsDistance(finalPoint, centroids[j]) < 1) {
                labels[i] = j;
                clusters[j].push_back(paths[i].front());
            }
        }

        // If the point has been assigned, store it in the clusters vector
        if (labels[i] != -1) {
            clusters[labels[i]].push_back(paths[i].front());
        }
    }
}

void removeLowCountClusters(std::vector<std::vector<cv::Point2f>>& clusters, std::vector<cv::Point2f>& centroids, int threshold) {
    for (int i=0; i<clusters.size(); i++) {
        if (clusters[i].size() < threshold) {
            clusters.erase(clusters.begin() + i);
            centroids.erase(centroids.begin() + i);
            i--;
        }
    }
}

void computeMean(std::vector<std::vector<cv::Point2f>>& clusters, std::vector<cv::Point2f>& means) {
    // For each cluster compute the geometric baricenter
    for (int i=0; i<clusters.size(); i++) {

        double meanX = 0;
        double meanY = 0;

        for (int j=0; j<clusters[i].size(); j++) {
            meanX += clusters[i][j].x;
            meanY += clusters[i][j].y;
        }

        meanX /= clusters[i].size();
        meanY /= clusters[i].size();

        means.push_back(cv::Point2f(meanX, meanY));
    }
}

double computeCovarianceScalar(std::vector<double> valuesA, std::vector<double> valuesB, double meanA, double meanB) {
    double covariance = 0;
    for (int i=0; i<valuesA.size(); i++) {
        covariance += (valuesA[i] - meanA) * (valuesB[i] - meanB);
    }
    covariance /= valuesA.size();

    return covariance;
}

void computeVarianceMatrices(std::vector<std::vector<cv::Point2f>>& clusters, std::vector<cv::Point2f>& means, std::vector<cv::Mat>& covMatrices) {
    for (int i=0; i<clusters.size(); i++) {
        // Extract the x and y values of the points in the cluster
        std::vector<double> valuesX, valuesY;
        for (int j=0; j<clusters[i].size(); j++) {
            valuesX.push_back(clusters[i][j].x);
            valuesY.push_back(clusters[i][j].y);
        }

        // Compute varX, varY, and covXY
        double meanX = means[i].x;
        double meanY = means[i].y;
        double varX = computeCovarianceScalar(valuesX, valuesX, meanX, meanX);
        double varY = computeCovarianceScalar(valuesY, valuesY, meanY, meanY);
        double covXY = computeCovarianceScalar(valuesX, valuesY, meanX, meanY);

        // Build the covariance matrix and push it in the output vector
        cv::Mat covMatrix(cv::Size(2,2), CV_64FC1);

        covMatrix.at<double>(0,0) = varX;
        covMatrix.at<double>(1,1) = varY;
        covMatrix.at<double>(0,1) = covXY;
        covMatrix.at<double>(1.0) = covXY;

        covMatrices.push_back(covMatrix);
    }
}

void getEllipse(std::vector<cv::Point2f>& means, std::vector<cv::Mat>& covMatrices, cv::Size axes, double angle);

cv::RotatedRect getErrorEllipse(double chisquare_val, cv::Point2f mean, cv::Mat covmat) {
	
	//Get the eigenvalues and eigenvectors
	cv::Mat eigenvalues, eigenvectors;
	cv::eigen(covmat, eigenvalues, eigenvectors);

	//Calculate the angle between the largest eigenvector and the x-axis
	double angle = atan2(eigenvectors.at<double>(0,1), eigenvectors.at<double>(0,0));

	//Shift the angle to the [0, 2pi] interval instead of [-pi, pi]
	if(angle < 0)
		angle += 6.28318530718;

	//Conver to degrees instead of radians
	angle = 180*angle/3.14159265359;

	//Calculate the size of the minor and major axes
	double halfmajoraxissize=chisquare_val*sqrt(eigenvalues.at<double>(0));
	double halfminoraxissize=chisquare_val*sqrt(eigenvalues.at<double>(1));

	//Return the oriented ellipse
	//The -angle is used because OpenCV defines the angle clockwise instead of anti-clockwise
	return cv::RotatedRect(mean, cv::Size2f(halfmajoraxissize, halfminoraxissize), angle);

}

bool isPointInsideEllipse(cv::Point2f point, cv::RotatedRect ellipse) {
    cv::Point2f center = ellipse.center;
    double axisX = ellipse.size.width / 2;
    double axisY = ellipse.size.height / 2;
    double angle = ( ellipse.angle) * CV_PI / 180;

    double deltaX = point.x - center.x;
    double deltaY = point.y - center.y;

    double temp1 = pow( (cos(angle)*deltaX + sin(angle)*deltaY), 2 ) / pow(axisX, 2);
    double temp2 = pow( (sin(angle)*deltaX - cos(angle)*deltaY), 2 ) / pow(axisY, 2);

    //printf("Temp: %f", temp1+temp2);
    
    return (temp1 + temp2) < 1;
}

void gaussianPruning(std::vector<std::vector<cv::Point2f>>& clusters, std::vector<cv::Point2f>& means, std::vector<cv::Mat>& covMatrices, double varianceThreshold, std::vector<cv::RotatedRect>& ellipses) {
    // Define the ellipses of each cluster and remove all the points outside of the ellipses
    for (int i=0; i<clusters.size(); i++) {
        ellipses.push_back( getErrorEllipse( varianceThreshold, means[i], covMatrices[i] ) );

        for (int j=clusters[i].size()-1; j>=0; j--) {
            if ( !isPointInsideEllipse( clusters[i][j], ellipses[i]) ) {
                //printf(" -> deleted!");
                clusters[i].erase(clusters[i].begin()+j);
            }
            //printf("\n");
        }
    }
}

void distancePruning(std::vector<std::vector<cv::Point2f>>& clusters, std::vector<cv::Point2f>& centroids, double distanceThreshold) {
    for (int i=0; i<clusters.size(); i++) {

        // Check if each point is within a certain distance. If not, discard it
        for (int j=clusters[i].size(); j>=0; j--) {

            if ( pointsDistance(clusters[i][j], centroids[i]) > distanceThreshold ) {
                clusters[i].erase(clusters[i].begin() + j);
            } 
        }
    }
}

double gaussianLikelihood(cv::Point2f p, cv::Point2f mean, cv::Mat covmat) {
    double det = cv::determinant(covmat);
    cv::Mat covInverse = covmat.inv();

    cv::Mat difference(cv::Size(1,2), CV_64FC1);
    difference.at<double>(0,0) = p.x - mean.x;
    difference.at<double>(1,0) = p.y - mean.y;

    cv::Mat temp = difference.t()*covInverse*difference;

    double likelihood = 1 / (2*CV_PI*sqrt(det)) * exp( -1/2 * temp.at<double>(0,0) );

    return likelihood;
}

void gaussianClustering(std::vector<cv::KeyPoint>& keypoints, std::vector<cv::Point2f>& means, std::vector<cv::Mat>& covMatrices, std::vector<std::vector<cv::Point2f>>& clusters) {
    
    clusters = std::vector<std::vector<cv::Point2f>>(means.size());

    // For each keypoint find the maximum likelihood and cluster it accordingly
    for (int i=0; i<keypoints.size(); i++) {
        double maxLikelihood = -1;
        int maxCluster = -1;

        for (int j=0; j<means.size(); j++) {
            double likelihood = gaussianLikelihood( keypoints[i].pt, means[j], covMatrices[j]);
            std::cout << likelihood << " ";

            if (likelihood >= maxLikelihood) {
                maxLikelihood = likelihood;
                maxCluster = j;
            }
        }

        std::cout << maxCluster << std::endl;
        clusters[maxCluster].push_back(keypoints[i].pt);
    }
}

void kmeansClustering(std::vector<cv::KeyPoint>& keypoints, std::vector<cv::Point2f>& centroids, std::vector<std::vector<cv::Point2f>>& clusters) {

    clusters = std::vector<std::vector<cv::Point2f>>(centroids.size());

    for (int i=0; i<keypoints.size(); i++) {

        double minDistance = 100000;
        double minIndex = -1;

        for (int j=0; j<centroids.size(); j++) {
            double distance = pointsDistance( keypoints[i].pt, centroids[j] );
            if ( distance < minDistance ) {
                minDistance = distance;
                minIndex = j;
            }
        }

        clusters[minIndex]. push_back(keypoints[i].pt);
    }
}