#include "HarrisCornerDetector.h"

Mat detectHarris(Mat img, float k)
{
	vector<vector<float>> gFilter = generateGaussianFilter(5, 1.0);
	img = conv(img, gFilter, -1, 1);

	Mat xGrad, yGrad, A, B, C;
	xGrad = conv(img, sobel_filter5x5_x, -1, 1);
	yGrad = conv(img, sobel_filter5x5_y, -1, 1);

	multiply(xGrad, xGrad, A, (1.0), CV_32SC1);
	A = conv(A, gFilter, -1, 1);
	multiply(xGrad, yGrad, B, (1.0), CV_32SC1);
	B = conv(B, gFilter, -1, 1);
	multiply(yGrad, yGrad, C, (1.0), CV_32SC1);
	C = conv(C, gFilter, -1, 1);

	Mat cornerResponse, det, traceSquare;
	Mat AB, C2;
	multiply(A, B, AB);
	multiply(C, C, C2);
	subtract(AB, C2, det);
	multiply(A + B, A + B, traceSquare);
	subtract(det, k * traceSquare, cornerResponse);

	cornerResponse = matIntToUchar(cornerResponse);

	threshold(cornerResponse, 200);
	nonMaxSuppression(cornerResponse, 5);

	return cornerResponse;
}

Mat matIntToUchar(Mat img)
{
	Mat result = Mat::zeros(img.size(), CV_8UC1);
	int max = 0;
	for (int y = 0; y < img.rows; ++y)
		for (int x = 0; x < img.cols; ++x)
			if (img.at<int>(y, x) > max)
				max = img.at<int>(y, x);
	for (int y = 0; y < img.rows; ++y)
		for (int x = 0; x < img.cols; ++x)
				result.at<uchar>(y, x) = saturate_cast<uchar>((float) img.at<int>(y, x) / max * 255);
	return result;
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

void threshold(Mat& img, int threshold)
{
	for (int y = 0; y < img.rows; ++y)
		for (int x = 0; x < img.cols; ++x)
			if (img.at<uchar>(y, x) < threshold)
				img.at<uchar>(y, x) = 0;
}