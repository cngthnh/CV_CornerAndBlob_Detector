#pragma once
#include "SIFT.h"

void matchBySIFT(Mat img1, Mat img2, int detector, string outFile);
Mat collectDescriptors(vector<CustomKeypoint>& keypoints);
vector<KeyPoint> convertKeypoints(vector<CustomKeypoint>& keypoints);