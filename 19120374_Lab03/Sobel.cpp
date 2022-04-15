#include "Sobel.h"

float _sobel_filter3x3_x[3][3] = { {-1,0,1},{-2,0,2},{-1,0,1} };
Mat sobel_filter3x3_x = Mat(3, 3, CV_32F, &_sobel_filter3x3_x);
float _sobel_filter3x3_y[3][3] = { {1,2,1},{0,0,0},{-1,-2,-1} };
Mat sobel_filter3x3_y = Mat(3, 3, CV_32F, &_sobel_filter3x3_y);