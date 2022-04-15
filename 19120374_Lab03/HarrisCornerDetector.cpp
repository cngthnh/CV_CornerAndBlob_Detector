#include "HarrisCornerDetector.h"

Mat detectHarris(Mat img, float k, float thresholdRatio)
{
	Mat gFilter = generateGaussianFilter(9, 1.0);
	img.convertTo(img, CV_32F);
	filter2D(img, img, -1, gFilter);

	Mat xGrad, yGrad, A, B, C;
	filter2D(img, xGrad, -1, sobel_filter3x3_x);
	filter2D(img, yGrad, -1, sobel_filter3x3_y);

	multiply(xGrad, xGrad, A, (1.0), CV_32F);
	filter2D(A, A, -1, gFilter);
	multiply(yGrad, yGrad, B, (1.0), CV_32F);
	filter2D(B, B, -1, gFilter);

	multiply(xGrad, yGrad, C, (1.0), CV_32F);
	filter2D(C, C, -1, gFilter);

	Mat cornerResponse, det, traceSquare;
	Mat AB, C2;
	multiply(A, B, AB);
	multiply(C, C, C2);
	subtract(AB, C2, det);
	multiply(A + B, A + B, traceSquare);
	subtract(det, k * traceSquare, cornerResponse);

	cornerResponse = convertAndSuppressNegatives(cornerResponse);
	double max, min;
	minMaxLoc(cornerResponse, &min, &max);

	threshold(cornerResponse, thresholdRatio * (max - min));
	nonMaxSuppression(cornerResponse, 5);

	return cornerResponse;
}

void suppress(Mat& img, int posX, int posY, int windowSize)
{
	int max = 0;
	for (int y = posY; y < posY + windowSize; ++y)
		for (int x = posX; x < posX + windowSize; ++x)
			if (img.at<uchar>(y, x) > max)
				max = img.at<uchar>(y, x);
	for (int y = posY; y < posY + windowSize; ++y)
		for (int x = posX; x < posX + windowSize; ++x)
			if (img.at<uchar>(y, x) < max)
				img.at<uchar>(y, x) = 0;
}

void nonMaxSuppression(Mat& img, int windowSize)
{
	for (int y = 0; y < img.rows - windowSize; ++y)
		for (int x = 0; x < img.cols - windowSize; ++x)
			suppress(img, x, y, windowSize);
}

void threshold(Mat& img, double threshold)
{
	for (int y = 0; y < img.rows; ++y)
		for (int x = 0; x < img.cols; ++x)
			if (img.at<uchar>(y, x) < threshold)
				img.at<uchar>(y, x) = 0;
}