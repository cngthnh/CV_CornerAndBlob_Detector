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
Mat generateGaussianFilter(int size, double sigma);
Mat generateLoGFilter(int size, double sigma);
Mat convertAndSuppressNegatives(Mat img);

vector<vector<uchar>> calcDirection(Mat xGrad, Mat yGrad);

void showCorners(Mat src, Mat features);