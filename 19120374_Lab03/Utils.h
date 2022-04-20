#pragma once
#define _USE_MATH_DEFINES
#include <string>
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cmath>

using namespace std;
using namespace cv;

class Extremum {
public:
	int x, y;
	int octave;
	int layer;
	Extremum(int y, int x, int octave, int layer);
};

class Octave {
public:
	int type;
	vector<Mat> scaledLayers;
	vector<Mat> blurredImages;
};

class CustomKeypoint : public KeyPoint {
public:
	int layer;
	int trueOctave;
	double scale;
	vector<double> descriptor;
};

char* getCmdOption(char** argv, int argc, const char* option);
bool cmdOptionExists(char** argv, int argc, const char* option);
Mat generateGaussianFilter(int size, double sigma);
Mat generateLoGFilter(int size, double sigma);
Mat convertAndSuppressNegatives(Mat img);

vector<vector<uchar>> calcDirection(Mat xGrad, Mat yGrad);

void showCorners(Mat src, Mat features, string outFile);

void drawKeypoints(Mat img, vector<CustomKeypoint> extrema);
int mod(int a, int b);

vector<double> flatten(vector<vector<vector<double>>>& matrix, int start, int end);

