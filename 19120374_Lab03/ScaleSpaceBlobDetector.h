#pragma once
#include "Utils.h"

enum BLOB_FILTER {
	LOG_FILTER,
	DOG_FILTER
};

Mat detectBlob(Mat img, int numberOfScales, int filterType = BLOB_FILTER::LOG_FILTER, float blobThreshold = 0.2, double initSigma = 1.0);
vector<Mat> calculateScaledLayers(Mat& img, int numberOfScales, int filterType, double initSigma = 1.0);
Mat detectExtrema(vector<Mat>& scaledLayers, pair<int, int> imageSize, float blobThreshold = 0.2, double initSigma = 1.0);
Mat detectDOG(Mat img, int numberOfScales, float blobThreshold = 0.2);
float findMax(Mat& scaled, int posX, int posY);
float findMin(Mat& scaled, int posX, int posY);
bool isBlob(vector<Mat> scaledLaplacian, int x, int y, float current);
void drawBlob(Mat src, Mat blobs, string outFile, double initSigma = 1.0);
vector<Mat> calculateLoGLayers(Mat& img, int numberOfScales, double initSigma = 1.0);
vector<Mat> calculateGaussianImages(Mat& img, int numberOfScales, double initSigma = 1.0);
vector<Mat> calculateDoGLayers(vector<Mat>& blurred);