#pragma once
#define _USE_MATH_DEFINES
#include <string>
#include <iostream>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cmath>

using namespace std;
using namespace cv;

char* getCmdOption(char** argv, int argc, const char* option);
bool cmdOptionExists(char** argv, int argc, const char* option);
vector<vector<float>> generateGaussianFilter(int size, float sigma);

vector<vector<uchar>> calcDirection(Mat xGrad, Mat yGrad);

template <class T>
int applyKernel(Mat& image, vector<vector<T>>& kernel, int startX, int startY) {
	int sum = 0;

	for (int y = 0; y < kernel.size(); ++y)
	{
		for (int x = 0; x < kernel[0].size(); ++x)
		{
			if (image.type() == CV_8UC1)
				sum += image.at<uchar>(y + startY, x + startX) * kernel[y][x];
			else
				sum += image.at<int>(y + startY, x + startX) * kernel[y][x];
		}
	}
	return sum;
}

template <class T>
Mat conv(Mat image, vector<vector<T>> kernel, int padding, int stride) {
	Mat result;
	Mat pad_image;

	if (image.type() != CV_8UC1 && image.type() != CV_32SC1)
	{
		cout << "Invalid input image\n";
		return result;
	}

	// "valid" padding
	if (padding == 0)
	{
		pad_image = image;
	}
	// "same" padding
	else if (padding == -1)
	{
		int padding_horiz = ceil((stride * (image.cols - 1) - image.cols + kernel[0].size()) / 2);
		int padding_vert = ceil((stride * (image.rows - 1) - image.rows + kernel.size()) / 2);
		copyMakeBorder(image, pad_image, padding_vert, padding_vert, padding_horiz, padding_horiz, BORDER_REPLICATE);
	}
	else
	{
		copyMakeBorder(image, pad_image, padding, padding, padding, padding, BORDER_REPLICATE);
	}

	result = Mat::zeros((pad_image.rows - kernel.size()) / stride + 1, (pad_image.cols - kernel[0].size()) / stride + 1, image.type());
	for (int y = 0; y < ceil((pad_image.rows - kernel.size()) / stride); ++y)
	{
		for (int x = 0; x < ceil(pad_image.cols - kernel[0].size() / stride); ++x)
		{
			if (image.type() == CV_8UC1)
				result.at<uchar>(y, x) = saturate_cast<uchar>(applyKernel(pad_image, kernel, x * stride, y * stride));
			else
				result.at<int>(y, x) = saturate_cast<int>(applyKernel(pad_image, kernel, x * stride, y * stride));
		}
	}

	return result;
}

void showFeatures(Mat src, Mat features);