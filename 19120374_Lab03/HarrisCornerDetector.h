#pragma once
#include "Utils.h"
#include "Sobel.h"

Mat detectHarris(Mat img, float k, float thresholdRatio);
void suppress(Mat& img, int posX, int posY, int windowSize);
void nonMaxSuppression(Mat& img, int windowSize);
void threshold(Mat& img, double threshold);
