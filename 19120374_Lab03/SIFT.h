#pragma once
#include "Utils.h"
#include "ScaleSpaceBlobDetector.h"

vector<CustomKeypoint> detectBySift(Mat& src, int detector, int numberOfScales);
CustomKeypoint* quadraticFitLocalization(int y, int x, int layerIndex, int octaveIndex, vector<Octave>& octaves, double contrastThreshold, int attempts = 5);
Mat computeHessianMatrix(int y, int x, int layerIndex, int octaveIndex, vector<Octave>& octaves);
Mat computeJacobianMatrix(int y, int x, int layerIndex, int octaveIndex, vector<Octave>& octaves);
vector<CustomKeypoint> getListOfKeypoints(vector<Octave> octaves, double initSigma = 1.0, double contrastThreshold = 0.05);
vector<CustomKeypoint> computeOrientations(CustomKeypoint keypoint, int octaveIndex, Mat gaussianImage, double radiusFactor = 3, int numberOfHistBins = 36, double peakRatio = 0.8, double scaleFactor = 1.5);
vector<CustomKeypoint> cleanKeypoints(vector<CustomKeypoint>& keypoints);
void generateDescriptors(vector<CustomKeypoint>& keypoints, vector<Octave>& octaves, int windowSize = 4, int numberOfHistBins = 8, int scaleMultiplier = 3, double maxDescriptorVal = 0.2);
bool isExtrema(vector<Mat>& scaledLaplacian, int x, int y, float current);