#include "sliceMatch.h"
#include<cmath>
#include <opencv2/core/types.hpp>
#include"Detector.hpp"
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;

void featurePruning(vector<Point2f>centers, vector<Point2f>points, vector<int>labels, vector<Point2f>&prunedPoints, vector<int>&newLabels, double threshold) {
	for (int i = 0; i < points.size(); i++) {
		for (int j = 0; j < centers.size(); j++) {
			if (labels[i] == j) {
				if (distance(centers[j], points[i]) < threshold) {
					prunedPoints.push_back(points[i]);
					newLabels.push_back(labels[i]);
				}
			}
		}
	}
}

float distance(Point2f a, Point2f b) {
	float dist = 0.0f;
	float dist_y = (a.y - b.y);
	float dist_x = (a.x - b.x);
	dist = dist_y*dist_y + dist_x*dist_x;
	dist = pow(dist,0.5);
	return dist;
}

void kMeans(vector<Point2f> centers, vector<Point2f>points, vector<int> &labels, int nCluster) {
	float temp_dist = 1000;
	float temp = 0;
	for (int i = 0; i < points.size(); i++) {
		for (int j = 0; j < centers.size(); j++) {
			temp = distance(centers[j], points[i]);
			if (temp < temp_dist) {
				labels[i] = j;
				temp_dist = temp;
			}
		}
		temp_dist = 1000;
	}
}

void clusterPruning(vector<Point2f>& centers, vector<int> labels,int threshold) {
	vector<int> counter(centers.size());
	for (int i = 0; i < labels.size(); i++) {
		counter[labels[i]]++;
	}
	vector<Point2f> newCenters;
	for (int i = 0; i < counter.size(); i++) {
		if (counter[i] >= threshold) {
			newCenters.push_back(centers[i]);
		}
	}
	centers = newCenters;
}

void imageSlicer(vector<vector<Point2f>> coordinatePoints,Mat originalImage,vector<Mat>& slicedImages) {
	//We get the coordinatesof the bounding box so that we can slice the image for each bounding box.
	int temp_x1, temp_x2, temp_y1, temp_y2;
	for (int i = 0; i < coordinatePoints.size(); i++) {
		temp_x1 = coordinatePoints[i][0].x;
		temp_y1 = coordinatePoints[i][0].y;
		temp_x2 = coordinatePoints[i][1].x - temp_x1;
		temp_y2 = coordinatePoints[i][1].y - temp_y1;
		slicedImages.push_back(originalImage(Rect(temp_x1,temp_y1,temp_x2,temp_y2)));
	}

}

void slicerViewer(vector<Mat>srcs, vector < vector<KeyPoint>>& keys, vector<Mat>& descriptors) {
	for (int i = 0; i < srcs.size(); i++) {
		vector<KeyPoint> temp;
		Mat temp2;
		keys.push_back(temp);
		descriptors.push_back(temp2);
		featureDetector(srcs[i], keys[i], descriptors[i]);
	}
}

void slicedFeatureViewer(vector<Mat> slicedImages,vector<vector<KeyPoint>>slicedKeypoints) {
	Scalar colorTab[] =
	{
		Scalar(0, 0, 255),
		Scalar(0,255,0),
		Scalar(255,100,100),
		Scalar(255,0,255),
		Scalar(0,255,255)
	};
	vector<vector<int>>slicedLabels(slicedImages.size());
	vector<vector<Point2f>>slicedCenters(slicedImages.size());
	vector<vector<Point2f>>slicedPoints(slicedImages.size());
	TermCriteria criteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 10, 1);
	for (int j = 0; j < slicedImages.size(); j++) {
		for (int i = 0; i < slicedKeypoints[j].size(); i++) {
			//cout << "here" << j<< " "<<i<< endl;
			slicedPoints[j].push_back(slicedKeypoints[j][i].pt);
		}
		kmeans(slicedPoints[j], 1, slicedLabels[j], criteria, 10, KMEANS_RANDOM_CENTERS, slicedCenters[j]);
	}
	//cout << (slicedPoints.size()==slicedImages.size()) << endl;
	for (int j = 0; j < slicedImages.size(); j++) {
		for (int i = 0; i < slicedCenters[j].size(); i++)
		{
			Point2f c = slicedCenters[j][i];
			circle(slicedImages[j], c, 40, colorTab[j], 1, LINE_AA);
		}
		for (int i = 0; i < slicedPoints[j].size(); i++) {
			Point2f c = slicedPoints[j][i];
			circle(slicedImages[j], c, 10, colorTab[j], 1, LINE_AA);
		}
	}
	for (int i = 0; i < slicedImages.size(); i++) {
		namedWindow(to_string(i), WINDOW_NORMAL);
		imshow(to_string(i), slicedImages[i]);
		waitKey(0);
	}
}

void classifier(Mat target,Point2f center, String& composition) {
	Mat cloned[3];
	split(target, cloned);
	//I'll use just the second channel since it is the one who brings the most information
	if ((meanCalculator(cloned[1], 9, center) >= 25 && meanCalculator(cloned[1], 9, center) <= 26)
		|| (meanCalculator(cloned[2], 9, center) >= 107 && meanCalculator(cloned[2], 9, center) <= 108)
		|| (meanCalculator(cloned[1], 9, center) >= 11 && meanCalculator(cloned[1], 9, center) <= 13)
		|| (meanCalculator(cloned[1], 9, center) >= 5 && meanCalculator(cloned[1], 9, center) <= 6)
		|| (meanCalculator(cloned[1], 9, center) >= 35 && meanCalculator(cloned[1], 9, center) <= 36)
		|| (meanCalculator(cloned[1], 9, center) >= 43 && meanCalculator(cloned[1], 9, center) <= 44)) {
		namedWindow("Salad", WINDOW_NORMAL);
		imshow("Salad", target);
		composition += "Salad, ";
		//cout << "value " << meanCalculator(cloned[1], 9, center) << endl;
	}
	else if ((meanCalculator(cloned[2], 9,center) >= 195 && meanCalculator(cloned[2], 9,center) <= 196)
		|| (meanCalculator(cloned[2], 9, center) >= 79 && meanCalculator(cloned[2], 9, center) <= 80) 
		|| (meanCalculator(cloned[2], 9, center) >= 126 && meanCalculator(cloned[2], 9, center) <= 126.5)) {
		namedWindow("bread", WINDOW_NORMAL);
		imshow("bread", target);
		//cout << "value " << meanCalculator(cloned[2], 9, center) << endl;
		composition += "Bread, ";
	}
	else if ((meanCalculator(cloned[2], 9, center) >= 130 && meanCalculator(cloned[2], 9, center) <= 131)) {
		namedWindow("Pasta Pesto", WINDOW_NORMAL);
		imshow("Pasta Pesto", target);
		//cout << "value " << meanCalculator(cloned[2], 9, center) << endl;
		composition += "Pasta Pesto, ";
	}
	else if ((meanCalculator(cloned[2], 9, center) >= 82 && meanCalculator(cloned[2], 9, center) <= 83)
		|| (meanCalculator(cloned[2], 9, center) >= 99 && meanCalculator(cloned[2], 9, center) <= 100)) {
		namedWindow("Pasta Tomato", WINDOW_NORMAL);
		imshow("Pasta Tomato", target);
		//cout << " value " << meanCalculator(cloned[2], 9, center) << endl;
		composition += "Pasta Tomato, ";
	}
	else if ((meanCalculator(cloned[2], 9, center) >= 118 && meanCalculator(cloned[2], 9, center) <= 119)) {
		namedWindow("Pasta Meat", WINDOW_NORMAL);
		imshow("Pasta Meat", target);
		composition += "Pasta Meat, ";
		//cout << "value " << meanCalculator(cloned[2], 9, center) << endl;
	}
	else if ((meanCalculator(cloned[2], 9, center) >= 75 && meanCalculator(cloned[2], 9, center) <= 76)
		|| (meanCalculator(cloned[2], 9, center) >= 86 && meanCalculator(cloned[2], 9, center) <= 87)
		|| (meanCalculator(cloned[2], 9, center) >= 122 && meanCalculator(cloned[2], 9, center) <= 123)) {
		namedWindow("Pasta with clums", WINDOW_NORMAL);
		imshow("Pasta with clums", target);
		composition += "Pasta with clums, ";
		//cout << "value " << meanCalculator(cloned[2], 9, center) << endl;
	}
	else if ((meanCalculator(cloned[2], 9, center) >= 141 && meanCalculator(cloned[2], 9, center) <= 142)) {
		namedWindow("Rice", WINDOW_NORMAL);
		imshow("Rice", target);
		composition += "Rice, ";
		//cout << "value " << meanCalculator(cloned[2], 9, center) << endl;
	}
	else if ((meanCalculator(cloned[2], 9, center) >= 60 && meanCalculator(cloned[2], 9, center) <= 63)) {
		namedWindow("Meat&Beans", WINDOW_NORMAL);
		imshow("Meat&Beans", target);
		composition += "Pork cutlet, Beans, ";
		//cout << "value " << meanCalculator(cloned[2], 9, center) << endl;
	}
	else if ((meanCalculator(cloned[2], 9, center) >= 26 && meanCalculator(cloned[2], 9, center) <= 27.5)) {
		namedWindow("Sea salad", WINDOW_NORMAL);
		imshow("Sea salad", target);
		composition += "Sea salad, Beans, Potatoes, ";
		//cout << "value " << meanCalculator(cloned[2], 9, center) << endl;
	}
	else if ((meanCalculator(cloned[2], 9, center) >= 128 && meanCalculator(cloned[2], 9, center) <= 129.5)) {
		namedWindow("Rabbit", WINDOW_NORMAL);
		imshow("Rabbit", target);
		composition += "Rabbit, ";
		//cout << "value " << meanCalculator(cloned[2], 9, center) << endl;
	}
	else if ((meanCalculator(cloned[2], 9, center) >= 132 && meanCalculator(cloned[2], 9, center) <= 133.5)) {
		namedWindow("Rabbit&Beans", WINDOW_NORMAL);
		imshow("Rabbit&Beans", target);
		composition += "Rabbit, Beans, ";
		//cout << "value " << meanCalculator(cloned[2], 9, center) << endl;
	}
	else if ((meanCalculator(cloned[2], 9, center) >= 138 && meanCalculator(cloned[2], 9, center) <= 139)
		|| (meanCalculator(cloned[2], 9, center) >= 129 && meanCalculator(cloned[2], 9, center) <= 130.5)
		|| (meanCalculator(cloned[2], 9, center) >= 122.5 && meanCalculator(cloned[2], 9, center) <= 124)) {
		namedWindow("FIsh&Potatoes", WINDOW_NORMAL);
		imshow("FIsh&Potatoes", target);
		composition += "Fish cutlet, Basil Potatoes, ";
		//cout << "value " << meanCalculator(cloned[2], 9, center) << endl;
	}
	else {
		namedWindow("Pasta with clums", WINDOW_NORMAL);
		imshow("Pasta with clums", target);
		composition += "Pasta with clums, ";
		//cout << "value " << meanCalculator(cloned[2],9,center) << endl;
	}
}

float meanCalculator(Mat target,int kSize,Point2f center) {
	float average=0;
	//if(center.x > target.cols || center.y > target.rows){}
	//else {
		for (int i = -kSize / 2; i < kSize / 2; i++) {
			for (int j = -kSize / 2; j < kSize / 2; j++) {
				average += target.at<uchar>(int(target.rows/2) + i, int(target.cols/2) + i);
			}
		}
	//}
	return average/(kSize*kSize);
}


void slideClassifier(Mat& target, int numIntervals,bool flag) {
	int sizeIntervalX = int(target.size().width / numIntervals);
	int sizeIntervalY = int(target.size().height / numIntervals);
	/*float counter = 0;
	float counter1 = 0;*/
	vector<float> numCl(13);//Bread,Pasta Pesto,Beans,Pork Cutlet
	const int offset = 0;
	vector<float> averages(3);
	vector<float> counters(13);//Bread,Pasta Pesto,Beans,Pork Cutlet
	vector<String> labels = { "Bread","Pasta Pesto","Beans","Pork Cutlet","Salad","Fish cutlet","potatoes",
	"Pasta tomato","Rabbit","Rice","Pasta Meat","Pasta Clums","Seafood salad"};
	for (int numWindowX = 0; numWindowX < numIntervals; numWindowX++) {
		for (int numWindowY = 0; numWindowY < numIntervals; numWindowY++) {
			for (int i = int((target.rows * numWindowX) / numIntervals); i < int((target.rows * (numWindowX + 1)) / numIntervals); i++) {
				for (int j = int((target.cols * numWindowY) / numIntervals); j <int((target.cols * (numWindowY + 1)) / numIntervals); j++) {
					averages[0] += target.at<Vec3b>(i, j)[0];
					//cout << int(target.at<Vec3b>(i, j)[0]) << endl;
					averages[1] += target.at<Vec3b>(i, j)[1];
					averages[2] += target.at<Vec3b>(i, j)[2];
				}
			}
			averages[0] = averages[0] / (sizeIntervalX * sizeIntervalY);
			averages[1] /= (sizeIntervalX * sizeIntervalY);
			averages[2] /= (sizeIntervalX * sizeIntervalY);
			//Bread 0,Pasta Pesto 1,Beans 2 ,Pork Cutlet 3,Salad 4 ,fish chutlet 5, potatoes 6 ,pasta tomato 7
			//Rabbit 8, Rice 9, Pasta Meat 10 ,Pasta Clums 11 , SeaFood salad 12
			if (averages[2] >= 186 && averages[2] <= 234			//Bread check
				&& averages[0] >= 71 && averages[0] <= 199)
				counters[0]++;
			if (averages[2] >= 62 && averages[2] <= 168				//Pesto check
				&& averages[0] >= 13 && averages[0] <= 96)
				counters[1]++;
			if(averages[2] >= 116 && averages[2] <= 180				//Beans check
				&& averages[0] >= 55 && averages[0] <= 148)
				counters[2]++;
			if(averages[2] >= 71 && averages[2] <= 153				//Pork check
				&& averages[0] >= 25 && averages[0] <= 122)
				counters[3]++;
			if (averages[2] >= 73 && averages[2] <= 235				//Salad check
				&& averages[0] >= 6 && averages[0] <= 77)
				counters[4]++;
			if (averages[2] >= 147 && averages[2] <= 243			//fishC check
				&& averages[0] >= 45 && averages[0] <= 200)
				counters[5]++;
			if (averages[2] >= 162 && averages[2] <= 236			//Potatoes check
				&& averages[0] >= 26 && averages[0] <= 147)
				counters[6]++;
			if (averages[2] >= 89 && averages[2] <= 221				//Tomato check
				&& averages[0] >= 1 && averages[0] <= 90)
				counters[7]++;
			if (averages[2] >= 89 && averages[2] <= 221				//Rabbit check
				&& averages[0] >= 53 && averages[0] <= 191)
				counters[8]++;
			if (averages[2] >= 91 && averages[2] <= 196				//Rice check
				&& averages[0] >= 35 && averages[0] <= 130)
				counters[9]++;
			if (averages[2] >= 53 && averages[2] <= 204				//PastaM check
				&& averages[0] >= 5 && averages[0] <= 175)
				counters[10]++;
			if (averages[2] >= 78 && averages[2] <= 227				//PastaC check
				&& averages[0] >= 4 && averages[0] <= 72)
				counters[11]++;
			if (averages[2] >= 72 && averages[2] <= 182				//SeaSalad check
				&& averages[0] >= 10 && averages[0] <= 243)
				counters[12]++;
			int temp_x1 = sizeIntervalX * (numWindowX);
			int temp_y1 = sizeIntervalY * (numWindowY);
			int temp_x2 = sizeIntervalX;
			int temp_y2 = sizeIntervalY;
			Mat fragment;
			fragment = target(Rect(temp_x1, temp_y1, temp_x2, temp_y2));
			//if (numCl[0]/(sizeIntervalX*sizeIntervalY) > 0.5) {
			//	/*namedWindow(to_string(numWindowX + 1) + " " + to_string(numWindowY + 1), WINDOW_NORMAL);
			//	imshow(to_string(numWindowX + 1) + " " + to_string(numWindowY + 1), fragment);*/
			//	counter++;
			//	/*waitKey(0);
			//	destroyWindow(to_string(numWindowX + 1) + " " + to_string(numWindowY + 1));*/
			//}
			//if (numCl[1] / (sizeIntervalX * sizeIntervalY) > 0.5) {
			//	/*namedWindow(to_string(numWindowX + 1) + " " + to_string(numWindowY + 1), WINDOW_NORMAL);
			//	imshow(to_string(numWindowX + 1) + " " + to_string(numWindowY + 1), fragment);*/
			//	counter1++;
			//	/*waitKey(0);
			//	destroyWindow(to_string(numWindowX + 1) + " " + to_string(numWindowY + 1));*/
			//}
			//numCl[0] = 0;
			//std::cout << "Blue average: " << averages[0] << endl;
			//std::cout << "Green average: " << averages[1] << endl;
			//std::cout << "Red average: " << averages[2] << endl;
			if (flag) {
				namedWindow(to_string(numWindowX + 1) + " " + to_string(numWindowY + 1), WINDOW_NORMAL);
				imshow(to_string(numWindowX + 1) + " " + to_string(numWindowY + 1), fragment);
				waitKey(0);
				destroyWindow(to_string(numWindowX + 1) + " " + to_string(numWindowY + 1));
			}
			averages[0] = 0;
			averages[1] = 0;
			averages[2] = 0;
		}
	}
	float totalInstances = 0;
	for (int i = 0; i < counters.size(); i++) {
		totalInstances += counters[i];
	}
	for (int i = 0; i < counters.size(); i++) {
		cout << "We found a " << float(counters[i] / totalInstances)*100 << " % of " << labels[i] << endl;
	}
	/*cout << "We found " << counters[0] << " instances of bread in this image" << endl;
	cout << "We found " << counters[1] << " instances of Pasta pesto in this image" << endl;
	cout << "We found " << counters[2] << " instances of beans in this image" << endl;
	cout << "We found " << counters[3] << " instances of pork in this image" << endl;
	cout << "We found " << counters[4] << " instances of salad in this image" << endl;
	cout << "We found " << counters[5] << " instances of fish in this image" << endl;
	cout << "We found " << counters[6] << " instances of potatoes in this image" << endl;
	cout << "We found " << counters[7] << " instances of pasta tomato in this image" << endl;
	cout << "We found " << counters[8] << " instances of rabbit in this image" << endl;
	cout << "We found " << counters[9] << " instances of rice in this image" << endl;
	cout << "We found " << counters[10] << " instances of Pasta Meat in this image" << endl;
	cout << "We found " << counters[11] << " instances of Pasta Clums in this image" << endl;
	cout << "We found " << counters[12] << " instances of SeaFood salad in this image" << endl;*/
}

cv::Mat lookupTable(int levels) {
	int factor = 256 / levels;
	cv::Mat table(1, 256, CV_8U);
	uchar* p = table.data;

	for (int i = 0; i < 128; ++i) {
		p[i] = factor * (i / factor);
	}

	for (int i = 128; i < 256; ++i) {
		p[i] = factor * (1 + (i / factor)) - 1;
	}

	return table;
}

cv::Mat colorReduce(const cv::Mat& image, int levels) {
	cv::Mat table = lookupTable(levels);

	std::vector<cv::Mat> c;
	cv::split(image, c);
	for (std::vector<cv::Mat>::iterator i = c.begin(), n = c.end(); i != n; ++i) {
		cv::Mat& channel = *i;
		cv::LUT(channel.clone(), table, channel);
	}

	cv::Mat reduced;
	cv::merge(c, reduced);
	return reduced;
}

//void slideCounter(Mat target, int numIntervals) {
//	int firstX = 0;
//	int firstY = 0;
//	int lastX = 0;
//	int lastY = 0;
//	int firstX2 = 0;
//	int firstY2 = 0;
//	int lastX2 = 0;
//	int lastY2 = 0;
//	bool firstFound = false;
//	bool firstFound2 = false;
//	int sizeIntervalX = target.size().width / numIntervals;
//	int sizeIntervalY = target.size().height / numIntervals;
//	for (int numWindowX = 0; numWindowX < numIntervals - 1; numWindowX++) {
//		for (int numWindowY = 0; numWindowY < numIntervals - 1; numWindowY++) {
//
//			int temp_x1 = sizeIntervalX * (numWindowX);
//			int temp_y1 = sizeIntervalY * (numWindowY);
//			int temp_x2 = sizeIntervalX;
//			int temp_y2 = sizeIntervalY;
//			Mat fragment;
//			fragment = target(Rect(temp_x1, temp_y1, temp_x2, temp_y2));
//			/*namedWindow(to_string(numWindowX + 1) + " " + to_string(numWindowY + 1), WINDOW_NORMAL);
//			imshow(to_string(numWindowX + 1) + " " + to_string(numWindowY + 1), fragment);*/
//			for (int i = temp_x1; i < temp_x2 * (numWindowX + 1); i++) {
//				for (int j = temp_y1; j < temp_y2 * ((numWindowY + 1)); j++) {
//					if ((40-offset <= target.at<Vec3b>(i, j)[0] && target.at<Vec3b>(i, j)[0] <= 150+offset &&
//						59 - offset <= target.at<Vec3b>(i, j)[1] && target.at<Vec3b>(i, j)[1] <= 160+offset &&
//						86 - offset <= target.at<Vec3b>(i, j)[2] && target.at<Vec3b>(i, j)[2] <= 176+ offset)) {
//							numCl[0]++;
//					}
//					if ((51 - offset <= target.at<Vec3b>(i, j)[0] && target.at<Vec3b>(i, j)[0] <= 95 + offset &&
//						63 - offset <= target.at<Vec3b>(i, j)[1] && target.at<Vec3b>(i, j)[1] <= 106 + offset &&
//						87 - offset <= target.at<Vec3b>(i, j)[2] && target.at<Vec3b>(i, j)[2] <= 133 + offset)) {
//						numCl[1]++;
//					}
//				}
//			}
//			cout << numCl[1] << endl;
//			cout << "this quadrant is " << numCl[1] / (sizeIntervalX * sizeIntervalY) << " % " << "beans" << endl;
//			if (numCl[0] / (sizeIntervalX * sizeIntervalY) > 0.5) {
//				if (!firstFound) {
//					firstFound = true;
//					firstX = numWindowX;
//					firstY = numWindowY;
//				}
//				/*namedWindow(to_string(numCl[0] / (sizeIntervalX * sizeIntervalY)) + "%", WINDOW_NORMAL);
//				imshow(to_string(numCl[0] / (sizeIntervalX * sizeIntervalY)) + "%", fragment);*/
//				lastX = numWindowX;
//				lastY = numWindowY;
//			}
//			if (numCl[1] / (sizeIntervalX * sizeIntervalY) > 0.5) {
//				if (!firstFound2) {
//					firstFound2 = true;
//					firstX2 = numWindowX;
//					firstY2 = numWindowY;
//				}
//				namedWindow(to_string(numCl[1] / (sizeIntervalX * sizeIntervalY)) + "%", WINDOW_NORMAL);
//				imshow(to_string(numCl[1] / (sizeIntervalX * sizeIntervalY)) + "%", fragment);
//				lastX2 = numWindowX;
//				lastY2 = numWindowY;
//			}
//			//waitKey(0);
//			numCl[0] = 0;
//			numCl[1] = 0;
//		}
//		/*waitKey(0);*/
//	}
//	Point2f ne = Point2f(firstX * sizeIntervalX, firstY * sizeIntervalY);
//	Point2f so = Point2f((lastX+1) * sizeIntervalX, (lastY+1) * sizeIntervalY);
//	rectangle(target, ne, so, Scalar(255, 100, 100), 3);
//	namedWindow("new BB", WINDOW_NORMAL);
//	imshow("new BB", target);
//	Point2f ne2 = Point2f(firstX2 * sizeIntervalX, firstY2 * sizeIntervalY);
//	Point2f so2 = Point2f((lastX2+1) * sizeIntervalX, (lastY2+1) * sizeIntervalY);
//	rectangle(target, ne2, so2, Scalar(100, 100, 100), 3);
//	namedWindow("new BB2", WINDOW_NORMAL);
//	imshow("new BB2", target);
//	/*destroyAllWindows();*/
//}

//const int dataSize = 5;
//vector<Mat> pasta;
//vector<Mat> meat;
//vector<Mat> salad;
//vector<Mat> potatoes;
//vector<Mat> bread;
//const int numClasses = 5;//Soon it will be 13 :)
//vector<float> scores(numClasses);
//vector<int> checks(numClasses);
//const vector<String> labels = {"Pasta","Meat","Salad","Potatoes","Bread"};
//vector<vector<Mat>> allClasses;
//void matcherInitializer() {
//	for (int i = 0; i < dataSize; i++) {
//		pasta.push_back(imread("../ourDataset/pasta" + to_string(i+1) + ".jpg"));
//		meat.push_back(imread("../ourDataset/carne" + to_string(i+1) + ".jpg"));
//		salad.push_back(imread("../ourDataset/insalata" + to_string(i+1) + ".jpg"));
//		potatoes.push_back(imread("../ourDataset/patate" + to_string(i+1) + ".jpg"));
//		bread.push_back(imread("../ourDataset/pane" + to_string(i+1) + ".jpg"));
//	}
//	allClasses.push_back(pasta);
//	allClasses.push_back(meat);
//	allClasses.push_back(salad);
//	allClasses.push_back(potatoes);
//	allClasses.push_back(bread);
//}

//void classifier(Mat target, vector<float>& personalScores) {
//	//The number 4 is gonna be changed into a constant soon
//	for (int classes = 0; classes < numClasses;classes++) {
//		for (int samples = 0; samples < numClasses; samples++) {
//			cicleMatcher(target,allClasses[classes][samples],classes);
//		}
//	}
//	//int area = target.rows*target.cols*0.01;
//	normalizeScores();
//	for (int i = 0; i < scores.size(); i++)
//		personalScores.push_back(scores[i]);
//	getScores();
//	bestScore();
//	cleanScores();
//}
//
//void findBestLabel(vector<vector<float>> allScores,int num) {
//	vector<float> bestLabels(num);
//	vector<int> classScores(num);
//	vector<int> imageItGotFound(num);
//	for (int i = 0; i < allScores[0].size(); i++) {
//		for (int j = 0; j < num; j++) {
//			if (bestLabels[i] < allScores[j][i]) {
//				bestLabels[i] = allScores[j][i];
//				classScores[i] = i;
//				imageItGotFound[i] = j;
//			}
//		}
//	}
//	for (int i = 0; i < num; i++) {
//		cout << "Best label for class " << classScores[i] << " is " << bestLabels[i] << " leading to label " << labels[classScores[i]] <<endl;
//		cout << "It got found in image " << imageItGotFound[i]<<endl;
//	}
//}
//
//void getScores() {
//	for (int i = 0; i < scores.size(); i++)
//		cout << labels[i] << " got a score of " << scores[i] << endl;
//}
//void cleanScores() {
//	for (int i = 0; i < scores.size(); i++) {
//		scores[i] = 0;
//	}
//}
//
//void normalizeScores() {
//	for (int i = 0; i < scores.size(); i++)
//		scores[i] = scores[i] / dataSize;
//}
//
//void bestScore() {
//	float currentMax = 0;
//	int index = 0;
//	for (int i = 0; i < scores.size(); i++) {
//		if (currentMax < scores[i] && checks[i]!=1) {
//			currentMax = scores[i];
//			index = i;
//		}
//	}
//	checks[index] = 1;
//	cout << "The subject contains: " << labels[index] << endl;
//	cout << "And it got a score of: " << currentMax << endl;
//}
//
//void cicleMatcher(Mat target, Mat candidate,int index) {
//	vector<KeyPoint> srcKeyPoints;
//	vector<KeyPoint> trgKeyPoints;
//	Mat srcDescriptors;
//	Mat trgDescriptors;
//	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_L1);
//	featureDetector(target, trgKeyPoints, trgDescriptors);
//	featureDetector(candidate, srcKeyPoints, srcDescriptors);			//This is gonna be changed with a constant array of images
//
//	//OpenCV tutorial
//	std::vector< std::vector<DMatch> > knn_matches;
//	matcher->knnMatch(trgDescriptors, srcDescriptors, knn_matches, 20);
//	const float ratio_thresh = 0.80f;
//	std::vector<DMatch> good_matches;
//	float numMatches = 0;
//	for (size_t i = 0; i < knn_matches.size(); i++)
//	{
//		if (knn_matches[i][0].distance <= ratio_thresh * knn_matches[i][1].distance)
//		{
//			good_matches.push_back(knn_matches[i][0]);
//			numMatches++;
//		}
//	}
//	scores[index] += numMatches;//trgKeyPoints.size();
//	//cout << numMatches << endl;
//}