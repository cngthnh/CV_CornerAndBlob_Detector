#pragma once
#include "Utils.h"

enum BLOB_FILTER {
	LOG_FILTER,
	DOG_FILTER
};

Mat detectBlob(Mat img, int numberOfScales, int filterType = BLOB_FILTER::LOG_FILTER, float blobThreshold = 0.2);
Mat detectDOG(Mat img, int numberOfScales, float blobThreshold = 0.2);
float findMax(Mat scaled, int posX, int posY);
bool isBlob(vector<Mat> scaledLaplacian, int x, int y, float current);
void drawBlob(Mat src, Mat blobs);